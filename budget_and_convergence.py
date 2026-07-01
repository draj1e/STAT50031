"""
budget_and_convergence.py
=========================
两个补充实验（对应施工方案 II.5 的 C4/C5 与论文 §6.2 证书、§7）：

(1) anytime 预算-质量曲线：给 CACG 递增的总预算 B，记录解质量 f(S)/greedy 与
    证书 L；验证"预算越多越好、且 L ≤ f(S) 始终成立"（近似证书有效性）。

(2) 成本感知指数收敛检查（C4）：对采样规则 σ̂/(c·n^p) 扫描 p ∈ {1.0,1.25,1.5,1.75,2.0}，
    看最优 p 附近总成本的相对变化是否 < 3%（边际收益递减 -> 判定"优化收敛"）。
"""
from __future__ import annotations
import math, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import make_instance
from submodular import NoisyMarginalOracle
from algorithms import exact_greedy, cacg, bai_greedy

RES = os.path.join(os.path.dirname(__file__), "results")
REGIME = dict(sigma_range=(0.2, 5.0), cost_range=(1.0, 30.0), cost_sigma_corr=-0.6, hard=False)


def budget_curve(seeds=range(12), n=100, universe=200, k=10, eta=0.75, delta=0.1):
    # 先测一次 CACG 的"自然收敛成本"作为预算刻度基准
    budgets_frac = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
    q_by_b = {b: [] for b in budgets_frac}
    cert_ok = 0
    cert_tot = 0
    for seed in seeds:
        inst = make_instance(seed=seed, n=n, universe_size=universe, k=k, **REGIME)
        g = exact_greedy(inst)
        # 自然收敛成本
        orc = NoisyMarginalOracle(inst, np.random.default_rng(seed))
        r_full = cacg(inst, orc, eta=eta, delta=delta)
        C = r_full.total_cost
        for bf in budgets_frac:
            orc = NoisyMarginalOracle(inst, np.random.default_rng(seed))
            r = bai_greedy(inst, orc, name="CACG-B", method="emp-subg", cost_aware=True,
                           eta=eta, delta=delta, budget=bf * C)
            q_by_b[bf].append(r.fval / g.fval)
            # 证书有效性：L ≤ f(S)
            cert_tot += 1
            cert_ok += int(r.certificate <= r.fval + 1e-9)
    xs = budgets_frac
    ys = [float(np.mean(q_by_b[b])) for b in xs]
    es = [float(np.std(q_by_b[b])) for b in xs]
    print("anytime 预算-质量：")
    for b, y in zip(xs, ys):
        print(f"  budget={b:>4}×C   f/greedy={y:.3f}")
    print(f"证书有效性 P(L≤f(S)) = {cert_ok}/{cert_tot} = {cert_ok/cert_tot:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar([b for b in xs], ys, yerr=es, marker="o", capsize=3, color="#d62728")
    ax.axhline(1 - 1/math.e, ls="--", color="gray", label="1-1/e")
    ax.set_xlabel("budget B (in units of CACG natural convergence cost C)")
    ax.set_ylabel("solution quality f(S) / greedy")
    ax.set_title("CACG anytime budget-quality curve (both-anti regime)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(RES, "fig_budget_curve.png")
    fig.savefig(p, dpi=140)
    print("saved", p)
    return dict(budgets=list(xs), quality=ys, cert_valid_rate=cert_ok / cert_tot)


def _cacg_p(inst, oracle, p, eta, delta):
    """带可调成本指数 p 的 CACG（用于收敛扫描）。"""
    import algorithms as A
    A._COST_EXP = p           # 由 algorithms 读取（下方 monkeypatch）
    return bai_greedy(inst, oracle, name=f"CACG-p{p}", method="emp-subg",
                      cost_aware=True, eta=eta, delta=delta)


def cost_exp_scan(seeds=range(12), n=100, universe=200, k=10, eta=0.75, delta=0.1):
    ps = [1.0, 1.25, 1.5, 1.75, 2.0]
    costs = {p: [] for p in ps}
    for seed in seeds:
        inst = make_instance(seed=seed, n=n, universe_size=universe, k=k, **REGIME)
        for p in ps:
            orc = NoisyMarginalOracle(inst, np.random.default_rng(seed))
            r = _cacg_p(inst, orc, p, eta, delta)
            costs[p].append(r.total_cost)
    means = {p: float(np.mean(costs[p])) for p in ps}
    best_p = min(means, key=means.get)
    best = means[best_p]
    print("\n成本感知指数 p 扫描（总成本，越低越好）：")
    for p in ps:
        print(f"  p={p:>4}   cost={means[p]:11.1f}   Δ vs best={100*(means[p]/best-1):+5.1f}%")
    # C4：best 邻域(±0.25)的相对变化
    neigh = [means[p] for p in ps if abs(p - best_p) <= 0.25 + 1e-9]
    spread = (max(neigh) - min(neigh)) / best
    print(f"best p={best_p}; 邻域相对波动={100*spread:.1f}% (C4 判据: <3% 视为收敛)")
    return dict(ps=ps, means=means, best_p=best_p, neighborhood_spread=spread)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    out = {}
    out["budget"] = budget_curve()
    out["cost_exp_scan"] = cost_exp_scan()
    with open(os.path.join(RES, "supp.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved", os.path.join(RES, "supp.json"))
