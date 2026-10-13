# 部署到云服务器

把 cortex 跑在云服务器上，别的电脑用浏览器就能访问，记忆也存在服务器上。

**架构**：两个容器，对外只开一个端口。

```
浏览器 ──HTTP :8000──► [caddy 容器]   ← 唯一入口，Basic 鉴权 + 反代
                          │
                          ▼
                       [cortex 容器]  ← 端口不发布，公网连不到
                          │
                          └── ./data  ← 记忆落盘（挂载到宿主机）
```

用 Docker 不只是为了「环境一致」。更实际的理由：本机是 Python 3.14，而
`chromadb` / `onnxruntime` 这类带原生扩展的包在 Linux 上的 3.14 wheel 很可能还没发布，
`pip install` 会退化成源码编译然后失败。镜像用 `python:3.12-slim` 一次性绕开这个问题。

---

## 一、本地准备

### 1. 确认敏感文件不会被提交

```bash
cd /path/to/cortex
git init
git add -A
git status --short | grep -E "\.env|cortex/"    # 应当只出现 .env.example
```

`.env`（真实 DeepSeek key）和 `.cortex/`（你的对话记忆）都已在 `.gitignore` 里。
**这一步不要跳过。**

### 2. 跑测试

```bash
./.venv/Scripts/python.exe -m pytest -q
```

必须写 `python -m pytest` 而不是裸 `pytest`：项目里没有 `conftest.py` /
`pyproject.toml`，裸跑时 pytest 不会把项目根放进 `sys.path`，会 `ModuleNotFoundError: agent`。

### 3. 本地先把镜像构建跑通

```bash
# 需要先启动 Docker Desktop（本机默认没在跑）
docker compose build
```

在本地排错比在服务器上方便得多。构建完可以顺手起一下验证：

```bash
docker compose up -d
# 浏览器开 http://localhost:8000  —— 注意本地这个 .env 还没有 CORTEX_AUTH_HASH，
# Caddy 会因为没有哈希而启动失败。本地验证只起 cortex 一个服务即可：
docker compose up -d cortex
```

### 4. 推到 GitHub 私有仓库

本机没装 `gh`，所以仓库要在网页上手动建：GitHub → New repository → **Private** → 不要勾
"Add a README"（否则和本地的首次提交冲突）。

```bash
git add -A
git commit -m "feat: 容器化部署（Dockerfile / compose / Caddy 反代 + Basic 鉴权）"
git branch -M main
git remote add origin https://github.com/<你的账号>/cortex.git
git push -u origin main
```

推送时会弹出浏览器让你登录 GitHub（Git Credential Manager），输一次即可。

**顺手确认历史里没有 key**：

```bash
git log --all -p | grep -i "sk-" || echo "OK：历史里没有 API key"
```

万一真的出现了，第一件事是去 DeepSeek 控制台**吊销那个 key**，而不是
`git rm` —— 提交历史里的东西删不干净。

---

## 二、服务器准备

```bash
ssh root@<公网IP>

# 装 Docker（官方脚本，Ubuntu / Debian 通用）
curl -fsSL https://get.docker.com | sh

# 让普通用户也能用 docker（重新登录后生效）
usermod -aG docker $USER

# 验证（必须看到 v2 插件，即 `docker compose` 而不是 `docker-compose`）
docker compose version
```

国内服务器建议配一下镜像加速，否则 `docker pull` 会很慢甚至超时：
在云厂商控制台里能拿到专属加速地址，写进 `/etc/docker/daemon.json` 的
`registry-mirrors` 后 `systemctl restart docker`。

### 拉代码

```bash
apt-get update && apt-get install -y git
git clone https://github.com/<你的账号>/cortex.git /opt/cortex
cd /opt/cortex
```

私有仓库需要凭据：简单做法是配一个 GitHub **只读 deploy key**（仓库 → Settings →
Deploy keys → Add，**不要勾 Allow write access**），比把 Personal Access Token
明文写进 `.git/config` 干净。

---

## 三、配置

### 1. 生成 Basic 鉴权密码哈希

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext '你的强密码'
# 输出形如： $2a$14$xxxxxxxxxxxxxxxxxxxxxx   ← 整串复制，含 $ 符号
```

### 2. 写 `.env`

```bash
cd /opt/cortex

cat > .env <<'EOF'
DEEPSEEK_API_KEY=sk-你的真实key
CORTEX_PORT=8000
CORTEX_AUTH_USER=cortex
CORTEX_AUTH_HASH=粘贴上一步的整串

# --- 运行预算（都可选，不设=不设限）---
# CORTEX_RUN_TIMEOUT=120       # 单次运行墙上时间上限（秒），不含等审批的时间
# CORTEX_MAX_TOKENS=50000      # 单次运行 token 上限（需 LLM 接口返回 usage 才生效）

# --- 成本核算（可选，两个都配才算钱）---
# 单位：元 / 百万 token
# CORTEX_PRICE_IN=2.0
# CORTEX_PRICE_OUT=8.0

# --- 审批闸门 ---
# 默认**开启**：write_file / run_command 执行前挂起等人点批准。
# CORTEX_APPROVAL=0            # 显式关掉（恢复「全部自动执行」的老行为）
# CORTEX_APPROVAL_SKIP=run_command   # 白名单：这些工具自动放行，不弹审批
EOF

chmod 600 .env
```

> `CORTEX_APPROVAL` 在本机跑 `run.bat` 时也是默认开的，所以本地和服务器行为一致。

注意是 `<<'EOF'`（带引号）：不带引号的话 shell 会把哈希里的 `$2a` 当变量展开，哈希就废了。

`docker-compose.yml` 和 `deploy/Caddyfile` 已经接好了：compose 把这两个变量注入
caddy 容器，Caddyfile 用 `{$CORTEX_AUTH_USER} {$CORTEX_AUTH_HASH}` 引用。
所以**这个文件里一个凭据都没有，可以安全提交进 git**。

> 别把哈希直接写进 Caddyfile 或 `docker-compose.yml`：后者会把 `$2a` 当变量插值，
> 必须写成 `$$2a$$14$$...` 才行。写在 `.env` 里没有这个问题（`.env` 的值不会被递归插值）。

顺带一提：`.env` 里如果少这两个变量，compose 会直接报错退出并告诉你缺哪个
（`:?` 语法），而不是起一个没有鉴权的服务。

### 3. 关于 `CORTEX_MEMORY_STORE`

不用改：`docker-compose.yml` 里写的是 `${CORTEX_MEMORY_STORE:-chroma}`，不设就是 chroma。
**如果反过来在 `.env` 里写了 `CORTEX_MEMORY_STORE=json`，服务器就会用 JSON 后端，
把迁过去的 chroma 记忆晾在一边** —— 所以 `.env.example` 里那一行是注释掉的。

---

## 四、迁移本地已有记忆

服务器上先把数据目录建好：

```bash
mkdir -p /opt/cortex/data
```

在**本地**把记忆传上去：

```bash
scp -r /path/to/cortex/.cortex/* root@<公网IP>:/opt/cortex/data/
```

回到服务器改属主——**这一步不能省**：

```bash
chown -R 10001:10001 /opt/cortex/data
```

容器里以 uid 10001 运行。权限不对的话写盘会失败，而且是**静默的**：
chroma 打不开会被 `memory/backend.py` 吞掉异常、降级回 JSON 后端，日志里只有一行警告，
现象是「记忆好像没迁上去」。改属主就避免了这件事。

> 更保险的做法是**先停掉容器再拷**，别让两个进程同时开着同一个 `chroma.sqlite3`：
> `cd /opt/cortex && docker compose stop cortex`（首次部署时容器还没起，可跳过）。

---

## 五、启动

```bash
cd /opt/cortex
docker compose up -d --build

docker compose ps                    # cortex 应当是 (healthy)
docker compose logs cortex | tail -20
```

**日志里必须出现这一行**：

```
[cortex] agent ready, memory=ChromaVectorMemory, tools=['calculator', 'list_directory', 'read_file', 'run_command', 'write_file']
```

两个自检点：

- `memory=ChromaVectorMemory` —— 不是 `VectorMemory`。出现 `VectorMemory` 说明
  chroma 没打开、静默降级了（多半是 `data/` 的属主不对）。
- 工具表里有 `list_directory` —— 那是**只可能来自 MCP** 的工具，它的存在证明
  MCP 子进程连上了。

### 开端口

云控制台的**安全组**放行 TCP 8000：

| 方向 | 协议 | 端口 | 源 |
|---|---|---|---|
| 入站 | TCP | 22 | 你的出口 IP |
| 入站 | TCP | 8000 | 0.0.0.0/0（或只放你自己的 IP，更安全） |
| 入站 | 其它 | —— | 拒绝 |

> **只开 `ufw` 是不够的**：Docker 发布端口是直接写 iptables 的 `DOCKER-USER` 链，
> 会绕过 ufw，所以 `ufw deny 8000` 对 `ports: 8000:8000` 不起作用。
> 云厂商的安全组才是真正生效的那一层边界。

---

## 六、验证

```bash
IP=<公网IP>
U=cortex; P='你的密码'

# 1. 没带凭据 → 必须 401（证明鉴权先于应用生效）
curl -s -o /dev/null -w "%{http_code}\n" http://$IP:8000/

# 2. 带凭据 → 307 跳到前端；health 里 has_llm 必须是 true
curl -s -u $U:$P http://$IP:8000/api/health

# 3. cortex 容器的端口没有发布到公网（应当连不上）
curl -s -m 5 http://$IP:8001/api/health || echo "OK：连不上"
```

`has_llm` 是 `false` 就说明 key 没注入进容器，检查 `.env` 里的
`DEEPSEEK_API_KEY`（尤其是有没有多写引号）。

### SSE 是不是真的流式（关键）

给每一行到达打上时间戳——被缓冲的话所有行会挤在同一秒：

```bash
curl -N -s -u $U:$P \
  "http://$IP:8000/api/chat/stream?message=算一下123*456&session_id=diag" \
  | while IFS= read -r line; do
      [ -n "$line" ] && printf '%s  %s\n' "$(date +%T)" "${line:0:100}"
    done
```

**判据**：第一行到最后的 `done` 之间要跨越好几秒，且多条 `answer_delta` 分布在不同的秒上。
如果全部集中在同一秒 → Caddy 在缓冲 → 检查 `deploy/Caddyfile` 里的 `flush_interval -1`
有没有写进去，改完 `docker compose restart caddy`。

### 记忆持久化

```bash
docker compose exec cortex ls -la /data /data/chroma    # 应看到 chroma.sqlite3
docker compose restart cortex
docker compose logs cortex | grep "agent ready"
```

重启后再问一件之前记过的事（**换一个 `session_id`**，这样短期记忆是空的，
能答出来只可能来自长期记忆）。

### 漏洞已堵（这次部署的回归验证）

```bash
docker compose exec cortex python -c "
from agent import build_agent
from agent.tools.base import ToolContext
a = build_agent(workspace_root='.')
try:
    print(a.registry.run('read_file', {'path': '.env'}, ToolContext(workspace_root='.')))
except ValueError as e:
    print('已拦住:', e)
"
```

或者更直接：在浏览器里跟 Agent 说「用 read_file 读一下 .env」，右列应当显示明确的拒绝。

> 说明这是**纵深防御的第二层**。第一层是 `.env` 根本不在容器里
> （`.dockerignore` 挡住了，密钥靠 compose 注入）。面试时可以这么讲：
> 黑名单是兜底，结构隔离才是主防线。

---

## 七、出事怎么查

| 现象 | 真实原因 | 怎么办 |
|---|---|---|
| 日志显示 `memory=VectorMemory` | chroma 打不开，静默降级 | `data/` 属主不对 → `chown -R 10001:10001 data`；或本地与容器的 chromadb 版本不一致 |
| 工具表里没有 `list_directory` | MCP 子进程没连上 | MCP 失败是**静默降级**的（工具凭空少几个）；确认 `mcp>=2.0` 装上了 |
| 回答一次性蹦出来，不流式 | 反代在缓冲 | Caddyfile 里的 `flush_interval -1` |
| 浏览器弹了登录框，输完还是 401 | `.env` 里的哈希坏了 | 哈希必须是整串含 `$`；写 `.env` 时用 `<<'EOF'` 单引号 heredoc |
| 点了发送没反应 | Basic 凭据没缓存成功 | 刷新页面重新登录一次（`EventSource` 不会为失败的请求再弹登录框） |
| 记忆写不进去 | 挂载目录不可写 | 同第一行的 chown；权限问题的表现全是「静默降级」 |

---

## 八、不要做的事

- **不要加 `--workers N`**，也不要用 gunicorn 多 worker、不要 `--scale cortex=2`。
  chroma 的 `PersistentClient` 同一目录只允许一个进程打开，多进程会互相抢锁。
  两个进程各开一个实例的结果是「记忆时有时无」，且日志里没有任何报错。
- **不要把 cortex 容器的端口 publish 出去**。它现在是 `expose`，只有 Caddy 能连到。
  一旦写成 `ports: "8000:8000"`，就等于绕过 Basic 鉴权把接口裸奔到公网。
- **不要把 `CORTEX_PORT` 之外的端口开给公网**，尤其是 8000 直接映射到 cortex 容器。

---

## 九、当前方案的已知边界

- **Basic over HTTP 是明文传密码**。`Authorization` 头只是 base64 编码不是加密，
  链路中间能还原出密码；对话内容同样是明文。缓解：安全组里把 8000 的源限制成
  你自己的出口 IP。
- **Basic 是单账号，不是多用户**。所有人共用一份长期记忆——短期记忆已经按
  `session_id` 隔离了（每个标签页一条），但长期记忆是全局一份。
- **没有限流**。凭据万一泄露，别人能刷你的 DeepSeek 余额。可以给 key 设额度上限。

### 下一步（已留好路）

**上域名 + HTTPS**：买到域名（国内服务器还需 ICP 备案）后，只改两处——
Caddyfile 把 `:8000` 换成 `cortex.example.com`（Caddy 会自动申请并续期
Let's Encrypt 证书），compose 里 caddy 的端口换成 `"80:80"` 和 `"443:443"`。
**应用代码一行都不用动**，这就是当初把鉴权放在反代而不是应用里的原因。
