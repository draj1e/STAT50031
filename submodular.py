"""
submodular.py
=============
最大覆盖（Max-Coverage）子模函数、精确边际预言机，以及带异方差噪声、异质成本的
噪声边际预言机（noisy marginal oracle）。

模型（对应论文 §2.3）：
    f(S) = | union_{j in S} G_j |,  其中 G_j 是全集 U 上的一个子集（"覆盖集"）。
    f 单调、子模、归一化 f(∅)=0。
    边际 Δ(e|S) = f(S ∪ {e}) - f(S) = |G_e \\ union_{j in S} G_j|。

噪声边际预言机（可重采样、独立、次高斯；对应论文 CACG 的良性区间）：
    一次查询返回  X = Δ(e|S) + ξ,  ξ ~ N(0, σ_{e,S}^2)（这里用高斯，属次高斯）。
    单次查询成本为 c_e（异质成本，元素相关）。

说明：σ、c 由 benchmark 生成并随实例携带；预言机只负责"按 σ 加噪、按 c 计费"。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CoverageInstance:
    """一个最大覆盖实例 + 噪声/成本注解。"""
    n: int                      # 候选集合(元素/臂)个数
    universe_size: int          # 全集 U 大小
    sets: list[np.ndarray]      # sets[e] = G_e，U 上被覆盖元素的下标数组
    k: int                      # 基数约束
    sigma: np.ndarray           # shape (n,)，每个元素的噪声标准差 σ_e（异方差）
    cost: np.ndarray            # shape (n,)，每个元素单次查询成本 c_e（异质成本）
    name: str = ""
    meta: dict = field(default_factory=dict)

    # ---- 位集合表示，便于快速求并/边际 ----
    def __post_init__(self):
        # 用 python int 位掩码表示每个覆盖集，求并/边际为按位或与 popcount，快速且精确
        self._mask = []
        for g in self.sets:
            m = 0
            for u in g:
                m |= (1 << int(u))
            self._mask.append(m)

    def coverage_mask(self, S) -> int:
        m = 0
        for e in S:
            m |= self._mask[e]
        return m

    def f(self, S) -> int:
        """精确目标值 f(S)=|覆盖并集|。"""
        return int(self.coverage_mask(S).bit_count())

    def marginal_exact(self, e: int, covered_mask: int) -> int:
        """精确边际 Δ(e|S)，其中 covered_mask 是当前集合 S 的覆盖掩码。"""
        return int((self._mask[e] & ~covered_mask).bit_count())

    def marginal_all_exact(self, covered_mask: int) -> np.ndarray:
        """对所有元素一次性返回精确边际（用于精确贪心/评判）。"""
        out = np.empty(self.n, dtype=np.int64)
        mask = self._mask
        for e in range(self.n):
            out[e] = (mask[e] & ~covered_mask).bit_count()
        return out


class NoisyMarginalOracle:
    """
    噪声边际预言机：query(e, covered_mask) 返回一次带噪观测，并累计成本与调用次数。
    - 期望等于真实边际 Δ(e|S)；
    - 噪声 ~ N(0, σ_e^2)，可重复独立采样；
    - 每次调用计费 c_e。

    为效率：同一轮内 covered_mask 固定，真实边际按 (covered_mask 版本, e) 缓存，
    重复查询只做一次 popcount，其余为 O(1) 加噪。调用 new_round(covered_mask) 切换轮。
    """
    def __init__(self, inst: CoverageInstance, rng: np.random.Generator):
        self.inst = inst
        self.rng = rng
        self.n_queries = 0          # 总查询次数
        self.total_cost = 0.0       # 总查询成本
        self.per_elem_queries = np.zeros(inst.n, dtype=np.int64)
        self._cur_mask = None
        self._marg_cache = {}

    def new_round(self, covered_mask: int):
        """进入新一轮（当前集合覆盖掩码固定）；重置边际缓存。"""
        self._cur_mask = covered_mask
        self._marg_cache = {}

    def query(self, e: int, covered_mask: int) -> float:
        if covered_mask != self._cur_mask:
            self.new_round(covered_mask)
        tm = self._marg_cache.get(e)
        if tm is None:
            tm = self.inst.marginal_exact(e, covered_mask)
            self._marg_cache[e] = tm
        x = tm + self.rng.normal(0.0, self.inst.sigma[e])
        self.n_queries += 1
        self.total_cost += float(self.inst.cost[e])
        self.per_elem_queries[e] += 1
        return float(x)

    def reset_counters(self):
        self.n_queries = 0
        self.total_cost = 0.0
        self.per_elem_queries[:] = 0


def true_marginal_range(inst: CoverageInstance) -> float:
    """边际取值范围界 R（用于经验-Bernstein 的范围项）。边际 ∈ [0, max|G_e|]。"""
    return float(max((len(g) for g in inst.sets), default=1))
