"""沙箱规则测试：路径越界 + 敏感文件黑名单。

黑名单不是理论需求：工作目录通常就是项目根目录，而 .env 正在里面。
只校验「路径没越界」时，read_file(".env") 会把 API key 原样返回，
公开部署时等于把密钥挂在首页。这组测试就是钉死这个行为。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent.tools import ToolContext, build_default_registry
from agent.tools.sandbox import resolve_in_workspace

ROOT = Path(__file__).resolve().parent.parent


# ---------- 敏感文件黑名单 ----------


def test_denies_env_file(tmp_path):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-real", encoding="utf-8")
    with pytest.raises(ValueError, match="敏感文件"):
        resolve_in_workspace(str(tmp_path), ".env")


def test_denies_env_variants(tmp_path):
    """`.env` 的各种变体同样可能装真密钥，不能只挡精确名。"""
    for name in (".env.local", ".env.production", ".env.bak"):
        with pytest.raises(ValueError, match="敏感文件"):
            resolve_in_workspace(str(tmp_path), name)


def test_denies_key_material(tmp_path):
    for name in ("server.key", "cert.pem", "id_rsa", "id_ed25519", "credentials"):
        with pytest.raises(ValueError, match="敏感文件"):
            resolve_in_workspace(str(tmp_path), name)


def test_deny_is_case_insensitive(tmp_path):
    """大小写不该成为绕过口子（Windows 上尤其）。"""
    for name in (".ENV", "ID_RSA", "Server.KEY"):
        with pytest.raises(ValueError, match="敏感文件"):
            resolve_in_workspace(str(tmp_path), name)


def test_allows_normal_project_files(tmp_path):
    """黑名单是「按文件名」而不是「按目录」，不能误伤正常源码阅读。"""
    (tmp_path / "README.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print(1)", encoding="utf-8")

    assert resolve_in_workspace(str(tmp_path), "README.md").name == "README.md"
    assert resolve_in_workspace(str(tmp_path), "notes.txt").name == "notes.txt"
    assert resolve_in_workspace(str(tmp_path), "src/main.py").name == "main.py"
    # .env.example 里没有密钥，但按前缀规则一起挡住了——这是有意的取舍，
    # 宁可严一点，免得哪天有人往里填了真 key。这条断言锁住这个决定。
    with pytest.raises(ValueError, match="敏感文件"):
        resolve_in_workspace(str(tmp_path), ".env.example")


# ---------- 路径越界 ----------


def test_blocks_traversal(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="越界"):
        resolve_in_workspace(str(tmp_path), "../outside.txt")


def test_blocks_sibling_with_common_prefix(tmp_path):
    """回归测试：曾经用 str.startswith 判断越界，于是 <root>-secret 会被放行。

    /tmp/ws-secret/x 的字符串前缀确实是 /tmp/ws，但它不是 /tmp/ws 的子路径。
    """
    root = tmp_path / "ws"
    root.mkdir()
    sibling = tmp_path / "ws-secret"
    sibling.mkdir()
    (sibling / "f.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="越界"):
        resolve_in_workspace(str(root), "../ws-secret/f.txt")


def test_allows_workspace_root_itself(tmp_path):
    assert resolve_in_workspace(str(tmp_path), ".") == tmp_path.resolve()


# ---------- 端到端：走真实工具 ----------


def test_read_file_tool_refuses_env(tmp_path):
    """对着真正的 read_file 打一发：应当拿到拒绝信息，且不含密钥内容。"""
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-realvalue", encoding="utf-8")
    (tmp_path / "safe.txt").write_text("hello", encoding="utf-8")

    registry = build_default_registry()
    ctx = ToolContext(workspace_root=str(tmp_path))

    refused = str(registry.run("read_file", {"path": ".env"}, ctx))
    assert "sk-realvalue" not in refused
    assert "拒绝" in refused

    # 同一个工具读正常文件仍然工作——证明拦的是文件性质，不是把工具关了
    assert "hello" in str(registry.run("read_file", {"path": "safe.txt"}, ctx))


def test_write_file_tool_refuses_env(tmp_path):
    """写侧同样要挡：不然可以把 .env 覆盖掉，或写个新的凭证文件。"""
    registry = build_default_registry()
    ctx = ToolContext(workspace_root=str(tmp_path))

    result = str(registry.run("write_file", {"path": ".env", "content": "x=1"}, ctx))
    assert "拒绝" in result
    assert not (tmp_path / ".env").exists()


def test_mcp_server_uses_same_denylist(tmp_path):
    """MCP 侧不能成为绕过口子——它复用同一套规则，不是自己抄一份。

    用子进程跑：server.py 是被当脚本拉起来的，模块级会固化 MCP_FS_ROOT，
    在当前进程里 import 拿不到干净的初始状态。
    """
    pytest.importorskip("mcp")
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-realvalue", encoding="utf-8")

    code = (
        "import os, sys\n"
        f"os.environ['MCP_FS_ROOT'] = {str(tmp_path)!r}\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from agent.mcp import server\n"
        "try:\n"
        "    server._resolve('.env')\n"
        "    print('LEAKED')\n"
        "except ValueError:\n"
        "    print('BLOCKED')\n"
        "print('workspace_root =', server.ROOT)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BLOCKED" in proc.stdout, proc.stdout


def test_live_registry_read_file_refuses_env(tmp_path, monkeypatch):
    """对着**实际运行时**的那个 read_file 打一发。

    为什么不能只测内置工具：`Agent.__init__` 先 `build_default_registry()`，
    再用 MCP 工具覆盖同名项（`ToolRegistry.register` 就是字典赋值），
    所以线上跑的是 **MCP 版本的 read_file**，内置那份反而是死代码。
    两个入口必须都被同一套沙箱规则盖住——这个测试锁的就是这件事。
    """
    pytest.importorskip("mcp")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "marker.txt").write_text("FROM-MCP-ROOT", encoding="utf-8")
    (workspace / ".env").write_text("DEEPSEEK_API_KEY=sk-leakvalue", encoding="utf-8")

    # 显式配置沙箱根。这里同时验证了它能被子进程读到——
    # 靠 os.environ 继承是**不行**的（mcp SDK 给子进程的是过滤过的白名单），
    # 必须由 build_agent 通过 env= 显式注入。
    monkeypatch.setenv("MCP_FS_ROOT", str(workspace))

    from agent import build_agent

    server = str(ROOT / "agent" / "mcp" / "server.py")
    agent = build_agent(
        mcp_command=sys.executable,
        mcp_args=[server],
        workspace_root=str(workspace),
    )

    # 前提：MCP 真的连上了（否则会退化成内置工具，这个测试就失去意义）
    assert "list_directory" in agent.registry.names(), "MCP 未连接，测试前提不成立"

    ctx = ToolContext(workspace_root=str(workspace))

    # 沙箱根确实传到了子进程：这个文件只在 workspace 里存在
    assert "FROM-MCP-ROOT" in str(agent.registry.run("read_file", {"path": "marker.txt"}, ctx))

    # 同一个工具读 .env 必须被拒，且不能吐出任何密钥内容
    refused = str(agent.registry.run("read_file", {"path": ".env"}, ctx))
    assert "sk-leakvalue" not in refused
