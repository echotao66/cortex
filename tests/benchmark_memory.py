"""记忆召回评测：召回准确率 + 时间衰减对「陈旧记忆」的抑制效果。

运行：python tests/benchmark_memory.py

两个实验，都只用本地计算、不联网：

1. 召回准确率（40 条记忆 / 20 条 query）
   10 条 query 与目标记忆共享词汇（字面匹配），10 条是改写问法（几乎不共享词汇）。
   对比两种 embedder：HashEmbedder（默认，零依赖）与 SentenceTransformerEmbedder（可选）。

2. 偏好变更场景（15 组）
   每组两条记忆都关于同一件事，但结论相反：旧的（120 天前）与新的（1 天前）。
   问「现在是什么情况」，看 top-1 落在新记忆还是旧记忆上。
   对比 lambda=0（纯余弦、不衰减）与 lambda=0.05（默认）。

   这一组是「时间衰减到底有没有用」的直接检验：如果衰减没用，
   新旧两条记忆的余弦相似度几乎一样，top-1 就是随机撞的。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 模型缓存目录如需指定，请先设置 HF_HOME 环境变量（不设则用 HuggingFace 默认缓存）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.memory.embedding import Embedder, HashEmbedder  # noqa: E402
from agent.memory.vector_memory import VectorMemory  # noqa: E402

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 40 条记忆：前 20 条是「关于用户的技术事实」，后 20 条是干扰项（同领域、易混）。
MEMORIES = [
    "用户熟悉 Python 和 FastAPI",
    "用户的项目 cortex 用 numpy 手写向量检索",
    "用户的项目 rag-engine 实现 BM25 与向量混合检索",
    "用户用 Docker Compose 部署项目",
    "用户用 Caddy 做反向代理和 Basic 鉴权",
    "用户用 DeepSeek 作为大模型",
    "用户用 pytest 写单元测试",
    "用户的服务器是 Ubuntu 22.04",
    "用户用 MCP 协议接入外部工具",
    "用户的记忆系统用余弦相似度加时间衰减",
    "用户的事实存储用三元组冲突合并",
    "用户的工具沙箱用命令白名单",
    "用户的路径沙箱按 Path.parents 判断越界",
    "用户用 SSE 推送事件流到前端",
    "用户的前端是原生 HTML 和 JS",
    "用户的 ReAct 循环最多 6 步",
    "用户的工具结果缓存 TTL 是 60 秒",
    "用户喜欢打羽毛球",
    "用户参加过蓝桥杯比赛",
    "用户的目标是拿到大厂后端实习 offer",
    # --- 干扰项：与上面的条目同领域、词汇高度重叠，但都「不回答」任何一条 query ---
    # 构造原则：换主体（同学的 / 读到的 / 听说过的），保证它跟目标记忆抢词、但不抢答案。
    # 这一条很关键：如果干扰项本身就是有效答案，标注就不成立、指标也就没意义了。
    "用户的同学熟悉 Java 和 Spring Boot",
    "用户读过一篇讲 Chroma 向量库的文章",
    "用户了解过纯关键词检索的原理",
    "用户只在教程里见过 Kubernetes 部署",
    "用户听说 Nginx 可以做静态资源托管",
    "用户调研过 GPT-4 的定价",
    "用户读过 unittest 的官方文档",
    "用户的实验室服务器是 CentOS 7",
    "用户看过 function calling 的协议说明",
    "用户对比过无衰减的纯余弦方案",
    "用户调研过用图数据库做事实存储",
    "用户知道 cat 命令能读任意文件",
    "用户读过字符串前缀判断越界导致漏洞的分析",
    "用户用过一次 WebSocket 的在线聊天 demo",
    "用户浏览过 Vue 的官方教程",
    "用户见过把循环上限设成 10 步的框架",
    "用户读过 TTL 设为 300 秒的缓存配置示例",
    "用户喜欢打篮球",
    "用户的队友参加过 ACM 校赛",
    "用户的室友的目标是先读研再就业",
]

# (query, 目标记忆下标, 类别)
QUERIES: list[tuple[str, int, str]] = [
    # --- 字面匹配：与目标记忆共享关键词 ---
    ("用户熟悉什么 Web 框架？", 0, "lexical"),
    ("用户的记忆系统怎么算分？", 9, "lexical"),
    ("用户的工具沙箱用什么机制？", 11, "lexical"),
    ("用户用什么部署项目？", 3, "lexical"),
    ("用户用什么做大模型？", 5, "lexical"),
    ("用户的服务器是什么系统？", 7, "lexical"),
    ("用户参加过什么比赛？", 18, "lexical"),
    ("用户用什么推送事件流？", 13, "lexical"),
    ("用户的缓存 TTL 是多少？", 16, "lexical"),
    ("用户的项目 rag-engine 用了什么检索？", 2, "lexical"),
    # --- 改写问法：几乎不与目标记忆共享词汇 ---
    ("他写后端用什么语言和框架？", 0, "paraphrase"),
    ("记忆检索的排序分数是怎么算出来的？", 9, "paraphrase"),
    ("怎么限制智能体能执行哪些命令？", 11, "paraphrase"),
    ("这个项目是怎么打包上线到服务器的？", 3, "paraphrase"),
    ("背后调用的模型是哪家的？", 5, "paraphrase"),
    ("云主机跑的是什么发行版？", 7, "paraphrase"),
    ("他拿过什么奖？", 18, "paraphrase"),
    ("前端怎么能实时看到 Agent 每一步？", 13, "paraphrase"),
    ("重复调用同一个工具会重复执行吗？", 16, "paraphrase"),
    ("把关键词检索和语义检索合起来用的是哪个项目？", 2, "paraphrase"),
]

# (旧记忆, 新记忆, 问「现在」的 query)
STALE_CASES: list[tuple[str, str, str]] = [
    ("用户用 Java 写后端", "用户现在主要用 Python 写后端", "用户用什么语言写后端？"),
    ("用户在准备考研", "用户已经保研，不再准备考研", "用户还在准备考研吗？"),
    ("用户喜欢喝咖啡", "用户戒了咖啡，改喝茶", "用户喜欢喝什么饮料？"),
    ("用户在北京工作", "用户已搬到上海工作", "用户现在在哪个城市工作？"),
    ("用户的实习岗位是算法岗", "用户转到了后端开发岗实习", "用户实习做什么岗位？"),
    ("用户的目标是考研到本校", "用户决定毕业后直接就业", "用户毕业后有什么打算？"),
    ("用户偏爱 Postgres", "用户的项目已迁移到 MySQL", "用户的项目用什么数据库？"),
    ("用户每天跑 5 公里", "用户膝盖受伤后改成游泳", "用户现在做什么运动？"),
    ("用户在学 Vue", "用户已经改用 React 做前端", "用户用什么前端框架？"),
    ("用户默认用 GPT-4", "用户的默认模型换成了 DeepSeek", "用户默认用哪个大模型？"),
    ("用户住在学校宿舍", "用户在校外租房住", "用户现在住在哪里？"),
    ("用户用 Windows 开发", "用户的主力机换成了 Mac", "用户用什么系统开发？"),
    ("用户的论文方向是计算机视觉", "用户的论文改做自然语言处理方向", "用户的论文是什么方向？"),
    ("用户喜欢用 VS Code", "用户现在用 PyCharm", "用户用什么编辑器？"),
    ("用户的实习时长是 3 个月", "用户实习延长到了 6 个月", "用户的实习要多久？"),
]

DAY = 86400.0


def load_embedders() -> list[tuple[str, Embedder]]:
    out: list[tuple[str, Embedder]] = [("hash", HashEmbedder())]
    try:
        from agent.memory.embedding import SentenceTransformerEmbedder

        out.append(("semantic", SentenceTransformerEmbedder(MODEL)))
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 语义 embedder 不可用（{exc}），只跑 HashEmbedder\n")
    return out


def build_memory(embedder: Embedder, with_distractors: bool = True) -> VectorMemory:
    """把记忆灌进去（全部按「刚刚写入」计）。

    with_distractors=False 时只放 20 条目标记忆，作为「干净库」基线；
    True 时再混入 20 条同领域干扰项，用来观察召回被稀释的程度。
    """
    mem = VectorMemory()
    texts = MEMORIES if with_distractors else MEMORIES[:20]
    for text in texts:
        mem.store(text, embedder.embed(text))
    return mem


def eval_recall(embedder: Embedder, k: int, *, with_distractors: bool = True) -> dict[str, float]:
    """返回 {'hit': 整体 hit@k, 'paraphrase': 改写类 hit@k}。"""
    mem = build_memory(embedder, with_distractors)
    hit = para_hit = 0
    n_para = 0
    for query, target, kind in QUERIES:
        got = [t for t, _ in mem.recall(embedder.embed(query), k=k)]
        ok = MEMORIES[target] in got
        hit += ok
        if kind == "paraphrase":
            n_para += 1
            para_hit += ok
    return {"hit": hit / len(QUERIES), "paraphrase": para_hit / n_para}


def eval_stale(embedder: Embedder, decay_lambda: float) -> float:
    """返回「新记忆排第一」的比例（越高越好）。

    旧记忆创建于 120 天前、新记忆创建于 1 天前。
    两条记忆问的是同一件事、结论相反，所以谁能排第一完全取决于打分策略。
    """
    now = time.time()
    hit = 0
    for old, new, query in STALE_CASES:
        mem = VectorMemory(decay_lambda=decay_lambda)
        mem.store(old, embedder.embed(old), created_at=now - 120 * DAY)
        mem.store(new, embedder.embed(new), created_at=now - 1 * DAY)
        top = mem.recall(embedder.embed(query), k=1)
        if top and top[0][0] == new:
            hit += 1
    return hit / len(STALE_CASES)


def main() -> None:
    embedders = load_embedders()

    print(f"记忆 {len(MEMORIES)} 条（含 20 条同领域干扰项）/ query {len(QUERIES)} 条")
    print(f"其中改写问法 {sum(1 for *_, k in QUERIES if k == 'paraphrase')} 条\n")

    print("【实验一】召回准确率（干净库 20 条 vs 混入干扰项 40 条）")
    print(
        f"{'embedder':>9} | {'库大小':>8} | {'hit@1':>7} | {'hit@3':>7} | {'改写 hit@1':>10}"
    )
    for name, emb in embedders:
        for label, with_dis in (("20 条", False), ("40 条", True)):
            m1 = eval_recall(emb, 1, with_distractors=with_dis)
            m3 = eval_recall(emb, 3, with_distractors=with_dis)
            print(
                f"{name:>9} | {label:>8} | {m1['hit']:>6.1%} | {m3['hit']:>6.1%} | "
                f"{m1['paraphrase']:>9.1%}"
            )

    print(f"\n【实验二】偏好变更：新记忆（1 天前）排第一的比例，共 {len(STALE_CASES)} 组")
    print(f"{'embedder':>9} | {'lambda=0（不衰减）':>18} | {'lambda=0.05（默认）':>18}")
    for name, emb in embedders:
        no_decay = eval_stale(emb, 0.0)
        decay = eval_stale(emb, 0.05)
        print(f"{name:>9} | {no_decay:>17.1%} | {decay:>17.1%}")


if __name__ == "__main__":
    main()
