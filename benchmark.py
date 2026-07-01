"""
benchmark.py
============
合成最大覆盖实例生成器（对应论文 §7 / 施工方案 II.4）。

可控维度：
  - n（候选集合数）、universe_size（全集）、k（基数约束）、cover_size（每个覆盖集大小范围）
  - 异方差 σ_e：log-uniform 分布，方差差异可调（sigma_spread）
  - 异质成本 c_e：log-uniform 分布，成本差异可调（cost_spread）；可与 σ 相关（cost_sigma_corr）
  - 难度：hard=True 时构造若干"边际非常接近"的候选（小 gap），压力测试样本量
所有随机性由传入的 seed 决定，保证可复现。
"""
from __future__ import annotations
import numpy as np
from submodular import CoverageInstance


def _log_uniform(rng, low, high, size):
    """在 [low, high] 上按对数均匀采样。"""
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def make_instance(
    seed: int,
    n: int = 100,
    universe_size: int = 200,
    k: int = 10,
    cover_size=(15, 40),
    sigma_range=(0.3, 6.0),          # 异方差范围；上限/下限比越大越"异方差"
    cost_range=(1.0, 20.0),          # 异质成本范围
    cost_sigma_corr: float = 0.0,    # ∈[-1,1]：+1 成本随σ增大，-1 反相关，0 独立
    hard: bool = False,              # 是否注入小 gap 的困难候选
    n_hard: int = 6,                 # 困难候选数量
    name: str = "",
) -> CoverageInstance:
    rng = np.random.default_rng(seed)

    # --- 覆盖集：为制造"边际递减"和差异，混合"热门元素"(popular)与随机元素 ---
    # 一部分全集元素是热门的（被很多集合覆盖 -> 冗余，体现子模性），其余稀有。
    n_popular = max(1, universe_size // 10)
    popular = np.arange(n_popular)
    sets = []
    for e in range(n):
        s = int(rng.integers(cover_size[0], cover_size[1] + 1))
        s = min(s, universe_size)
        # 约 40% 名额给热门元素（制造重叠/冗余），其余随机
        n_pop = int(round(0.4 * s))
        n_pop = min(n_pop, n_popular)
        part_pop = rng.choice(popular, size=n_pop, replace=False) if n_pop > 0 else np.array([], dtype=int)
        part_rand = rng.choice(np.arange(universe_size), size=s - n_pop, replace=False)
        g = np.unique(np.concatenate([part_pop, part_rand]).astype(int))
        sets.append(g)

    # --- 异方差 σ_e ---
    sigma = _log_uniform(rng, sigma_range[0], sigma_range[1], n)

    # --- 异质成本 c_e（可与 σ 相关）---
    base_cost = _log_uniform(rng, cost_range[0], cost_range[1], n)
    if abs(cost_sigma_corr) > 1e-9:
        # 用 σ 的秩来诱导相关性：corr>0 时 σ 大者成本也大
        order = np.argsort(np.argsort(sigma))  # σ 的秩 0..n-1
        rank_cost = np.sort(base_cost)
        corr_cost = rank_cost[order] if cost_sigma_corr > 0 else rank_cost[::-1][order]
        w = abs(cost_sigma_corr)
        cost = (1 - w) * base_cost + w * corr_cost
    else:
        cost = base_cost

    inst = CoverageInstance(
        n=n, universe_size=universe_size, sets=sets, k=k,
        sigma=sigma.astype(float), cost=cost.astype(float),
        name=name or f"cov_n{n}_U{universe_size}_k{k}_seed{seed}",
        meta=dict(seed=seed, hard=hard, cost_sigma_corr=cost_sigma_corr),
    )

    # --- 困难实例：让若干候选在"空集下"的边际几乎相等（小 gap）---
    if hard:
        # 取一个基准覆盖集大小 base_s，构造 n_hard 个大小几乎相同、彼此几乎不重叠的集合，
        # 使它们在第一轮的边际（≈各自大小）非常接近，制造 best-arm 的小 gap。
        base_s = int(np.median([len(g) for g in sets]))
        # 分配互不相交的随机元素块，保证边际≈base_s 且彼此独立覆盖
        pool = rng.permutation(universe_size)
        idx = 0
        for h in range(min(n_hard, n)):
            size_h = base_s + int(rng.integers(-1, 2))  # 差 ±1，gap 很小
            size_h = max(1, min(size_h, universe_size - idx))
            block = pool[idx: idx + size_h]
            idx += size_h
            if idx >= universe_size:
                break
            inst.sets[h] = np.unique(block.astype(int))
        # 重新构建位掩码
        inst.__post_init__()
    return inst


def optimal_bruteforce(inst: CoverageInstance, max_n_for_bruteforce: int = 18):
    """小实例可暴力求 OPT（用于校验近似比）；否则返回 None。"""
    from itertools import combinations
    if inst.n > max_n_for_bruteforce:
        return None
    best = 0
    for combo in combinations(range(inst.n), min(inst.k, inst.n)):
        best = max(best, inst.f(combo))
    return best


if __name__ == "__main__":
    inst = make_instance(seed=0, n=12, universe_size=60, k=4)
    print("instance:", inst.name)
    print("cover sizes:", [len(g) for g in inst.sets])
    print("sigma (min/median/max): %.2f / %.2f / %.2f" %
          (inst.sigma.min(), np.median(inst.sigma), inst.sigma.max()))
    print("cost  (min/median/max): %.2f / %.2f / %.2f" %
          (inst.cost.min(), np.median(inst.cost), inst.cost.max()))
    opt = optimal_bruteforce(inst)
    print("OPT (bruteforce) =", opt)
