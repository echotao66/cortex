"""Consolidator：固化式记忆（对标 picoagents 的 dual_memory.consolidate）。

思路：不是把整段对话原样存起来，而是让 LLM 把对话「压缩」成两部分——
1. facts   ：结构化事实三元组（写进 FactStore）
2. summary ：一段摘要（作为一条新记忆写进 VectorMemory）

这样记忆是「提炼过的」，而不是「堆砌的」，检索和查询都更干净。
"""

from __future__ import annotations

import json

from ..llm import LLMClient

SYSTEM_PROMPT = """你是一个记忆固化器。从对话中提取稳定的事实，并写一段简短摘要。

只输出一个 JSON 对象，不要输出任何其他文字，也不要包在 markdown 代码块里。格式：
{"facts": [{"subject": "...", "predicate": "...", "object": "..."}], "summary": "..."}

规则：
1. 只提取明确、稳定的信息（人名、偏好、位置、重要决定等），忽略闲聊。
2. subject 一般填「用户」，谓词用「住在/喜欢/叫/是」这类简洁动词。
3. 没有值得记的事实时，facts 返回空数组。
4. summary 用 2-3 句中文概括这段对话的要点。
"""


class Consolidator:
    """用 LLM 把原始对话固化成事实 + 摘要。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def consolidate(self, conversation: str) -> dict:
        """返回 {"facts": [{"subject","predicate","object"}, ...], "summary": str}。"""
        content = self.llm.chat(SYSTEM_PROMPT, f"对话内容：\n{conversation}")
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """容错解析：剥离模型可能误加的 markdown 代码块围栏。"""
        text = content.strip()
        if text.startswith("```"):
            text = text.lstrip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.rstrip("`").strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("固化结果必须是 JSON 对象")
        data.setdefault("facts", [])
        data.setdefault("summary", "")
        return data
