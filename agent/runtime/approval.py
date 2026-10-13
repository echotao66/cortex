"""审批策略：哪些工具调用必须先经人工确认才能执行。

agent 拿到 shell 和文件写权限之后，「模型自己决定要不要执行」这件事本身
就是最大的风险面 —— prompt 注入、幻觉参数、误判意图，任何一个都能让它
去 `rm` 或者覆写文件。审批闸门把「决定」和「执行」拆开：模型提出调用，
人点头才真的落地。

策略与工具分离：工具只**声明**风险（`Tool.requires_approval`），
要不要拦、拦哪些，统一由这里的策略决定 —— 否则策略会散落在各工具里，
上线时想按环境调整（本地放开、生产收紧）就没法改。

默认 disabled。演示/生产在装配点显式打开，单测与库调用方不受影响。
"""

from __future__ import annotations

from typing import Any, Iterable

# 开箱即用的拦截名单：不可逆、能改动真实环境的操作。
DEFAULT_REQUIRE = frozenset({"write_file", "run_command"})

# 理由文本给用户看，不是给日志看 —— 要能让人一眼判断该不该点「批准」。
REASON_IRREVERSIBLE = "「{name}」是写入 / 执行类操作，结果不可逆"
REASON_DECLARED = "工具「{name}」声明自己需要人工确认"


class ApprovalPolicy:
    """enabled=False 时 check() 恒返回 None（全放行），等价于没有闸门。

    require      ：强制要审批的工具名（默认 write_file / run_command）
    auto_approve ：白名单，优先级最高 —— 用于「我就是要让它自动跑某条命令」
                   的受控放行，而不是把整个闸门关掉
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        require: Iterable[str] | None = None,
        auto_approve: Iterable[str] = (),
        max_rejections: int = 3,
    ) -> None:
        self.enabled = enabled
        self.require = (
            frozenset(require) if require is not None else DEFAULT_REQUIRE
        )
        self.auto_approve = frozenset(auto_approve)
        # 被拒够多次就不再拦，直接把「此路不通」告诉模型。没有这个上限，
        # 模型可以合法地反复提议同一个操作、用户反复拒绝，循环永不收敛。
        self.max_rejections = int(max_rejections)

    @classmethod
    def disabled(cls) -> "ApprovalPolicy":
        return cls(enabled=False)

    def check(self, name: str, args: dict[str, Any], *, declared: bool = False) -> str | None:
        """返回需要人工审批的理由；None 表示放行。

        declared 是调用方从 Tool.requires_approval 读到的自我声明 ——
        这样策略层不必依赖 tools 包，保持单向依赖。
        """
        if not self.enabled or name in self.auto_approve:
            return None
        if name in self.require:
            return REASON_IRREVERSIBLE.format(name=name)
        if declared:
            return REASON_DECLARED.format(name=name)
        return None
