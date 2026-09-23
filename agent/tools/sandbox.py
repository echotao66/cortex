"""执行安全边界：路径沙箱 + 命令沙箱。

说明：这是教学用的「安全边界」演示——说清楚哪些该拦、怎么拦。
真正的隔离要靠容器/虚拟机，这里做的是「尽量缩小模型乱来的破坏面」。
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

# 只允许执行这些只读/无害命令（按命令首词白名单）。
# 注意：故意不含 cat/head/tail 这类能读任意文件的命令，
# 文件读取走带路径校验的 read_file 工具，避免绕过路径沙箱。
ALLOWED_COMMANDS = {"ls", "pwd", "echo", "whoami", "date"}

# 包含这些子串的命令一律拒绝（重定向、管道、危险命令等）。
BLOCKED_PATTERNS = [
    "rm", "del", "format", "shutdown", "reboot", "sudo",
    "chmod", "chown", "mkfs", "dd",
    ">", "|", ";", "&", "$(", "`",
]

# 即使落在工作目录内，也不允许碰的凭证类文件。
#
# 为什么单靠路径沙箱不够：工作目录通常就是项目根目录，而 .env 正在里面。
# 只校验「路径没越界」时，Agent 一句 read_file(".env") 就能把 API key
# 原样吐回来，而且会作为 tool_result 事件显示在前端——公开部署时等于把
# 密钥挂在首页。所以这里补一道按文件名的黑名单。
#
# 刻意用「文件名精确匹配 / 前缀 / 后缀」而不是路径规则，这样项目源码
# （.py / .md / .html）全不受影响，「让 Agent 读自己的代码」的演示保得住。
DENY_NAMES = {
    ".env", ".netrc", ".npmrc", ".git-credentials", ".htpasswd",
    "id_rsa", "id_ed25519", "credentials", "secrets",
}
# .env 的各种变体（.env.local / .env.production …）同样可能装真密钥，
# 这条会连 .env.example 一起挡住。示例文件本身没密钥，挡了不损失什么；
# 但万一有人往里面填了真 key 就来不及了，所以宁可严一点。
DENY_PREFIXES = (".env.",)
DENY_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".jks", ".keystore")


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in DENY_NAMES
        or name.startswith(DENY_PREFIXES)
        or name.endswith(DENY_SUFFIXES)
    )



def resolve_in_workspace(workspace_root: str, rel_path: str) -> Path:
    """把相对路径解析到工作目录内，并拒绝敏感文件。

    两道关：① 路径不得越出工作目录；② 即使在工作目录内，凭证类文件也不给碰。
    """
    root = Path(workspace_root).resolve()
    candidate = (root / rel_path).resolve()
    # 这里曾经是 str(candidate).startswith(str(root))——纯字符串前缀比较，
    # 于是 /opt/cortex-secret 会被判定成 /opt/cortex 的子路径而放行。
    # 改成按路径层级判断（Path.parents），/opt/cortex-secret 不再匹配。
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"路径越界：只允许访问工作目录内的文件（{rel_path}）")
    if _is_sensitive(candidate):
        print(f"[cortex] 已拒绝访问敏感文件：{candidate.name}", file=sys.stderr)
        raise ValueError(f"拒绝访问敏感文件（{candidate.name}）：凭证类文件不对外开放")
    return candidate


def validate_command(command: str) -> str:
    """校验命令是否安全，返回规范化后的命令；不安全则抛 ValueError。"""
    cmd = command.strip()
    if not cmd:
        raise ValueError("命令不能为空")

    lowered = cmd.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            raise ValueError(f"命令包含危险片段「{pattern}」，已拒绝")

    parts = shlex.split(cmd)
    if not parts:
        raise ValueError("命令不能为空")
    if parts[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"命令「{parts[0]}」不在白名单 {sorted(ALLOWED_COMMANDS)} 中")
    return cmd
