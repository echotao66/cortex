"""可插拔的 Embedder 层。

为什么单独抽象这一层：DeepSeek 官方 API 目前**不提供 embedding 接口**，
所以「对话/固化」走 DeepSeek，而「向量化」在这里解耦出去——
这正是 picoagents 里 chat provider 与 embedding provider 分离的思路。

默认用 HashEmbedder（定性、离零依赖、确线可跑），
生产/想要真语义时可换成 SentenceTransformerEmbedder 或任意 embedding API。
"""

from __future__ import annotations

import re
import zlib
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """把文本映射成固定维度的向量。所有实现必须保证同一文本产出相同向量。"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """返回 L2 归一化后的向量（利于用余弦相似度做检索）。"""

    @abstractmethod
    def dim(self) -> int:
        """返回向量维度。"""


class HashEmbedder(Embedder):
    """零依赖的确定性 embedding：hashing trick + 词频统计。

    不是真正语义化的（"苹果手机"和"苹果水果"可能得到相近向量），
    但足够演示「向量检索 + 时间衰减」的完整流程，且不联网、可复现。
    面试时可以直接说：这是可插拔的一层，生产环境替换实现即可。
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须 > 0")
        self._dim = int(dim)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in self._tokenize(text):
            # 必须用 crc32 而不是内置 hash：内置 hash 对 str 被 PYTHONHASHSEED
            # 加盐，跨进程会给出不同下标，于是「进程 A 落盘的向量」和
            # 「进程 B 算出的 query 向量」落在两个不相干的哈希空间里，
            # 余弦相似度退化成噪声、落盘的记忆再也检索不到。
            idx = zlib.crc32(token.encode("utf-8")) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)# L2归一化:将向量转化为单位向量，方便后续余弦相似度计算
        if norm > 0:
            vec /= norm
        return vec

    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """英文按词、中文按字 + bigram，兼顾两种语言。
        不完美，但零依赖、够用，是刻意留的替换点。"""
        tokens: list[str] = []
        tokens += re.findall(r"[a-z0-9]+", text.lower())
        cjk = re.findall(r"[一-鿿]", text)
        tokens += cjk
        for i in range(len(cjk) - 1):
            tokens.append(cjk[i] + cjk[i + 1])
        return tokens


class SentenceTransformerEmbedder(Embedder):
    """本地语义 embedding（可选，需额外依赖）。

    安装：pip install sentence-transformers
    国内下载模型建议先设置：export HF_ENDPOINT=https://hf-mirror.com
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 sentence-transformers：pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def dim(self) -> int:
        return self._dim
