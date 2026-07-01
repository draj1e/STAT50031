"""
run_experiments.py
==================
实验驱动（对应论文 §7 / 施工方案 II.4, II.5）。

对每个"regime"（噪声/成本配置）跑多 seed，比较算法：
    exact_greedy (参照上界), fixed_sampling, confidence_greedy, CACG, 及 2×2 消融
指标：
    f 值、与精确贪心差距、总查询次数、总查询成本、经验失败率、证书 L
"经验失败率"定义：算法所选集合 S 的 f(S) < (1-1/e)·OPT_greedy - eps 的实例比例，
其中 OPT_greedy 用精确贪心值作代理上界（合成小实例也可用真 OPT）。

用法：
    python run_experiments.py            # 跑默认全套，存 results/summary.json
    python run_experiments.py --quick    # 少 seed 快速版
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import asdict
import numpy as np

from benchmark import make_instance, optimal_bruteforce
from submodular import NoisyMarginalOracle
from algorithms import (exact_greedy, fixed_sampling_greedy, fixed_sampling_guaranteed,
                        confidence_greedy, cacg, cacg_no_eb, cacg_no_cost, cacg_mp,
                        random_baseline)

# ---- 各 regime：控制异方差程度与异质成本程度 ----
# σ 上界适中(≤5)以让"有保证"的基线也能在样本上限内收敛，避免截断伪影 -> 成本可比。
REGIMES = {
    # 温和：方差、成本差异都不大
    "mild":        dict(sigma_range=(0.5, 1.5),  cost_range=(1.0, 3.0),  cost_sigma_corr=0.0,  hard=False),
    # 强异方差：σ 跨度大（经验方差自适应应发挥作用）
    "hetero-var":  dict(sigma_range=(0.2, 5.0),  cost_range=(1.0, 3.0),  cost_sigma_corr=0.0,  hard=False),
    # 强异质成本：c 跨度大（成本感知应发挥作用）
    "hetero-cost": dict(sigma_range=(0.5, 1.5),  cost_range=(1.0, 30.0), cost_sigma_corr=0.0,  hard=False),
    # 双强 + 负相关（σ 大者成本反而低：成本感知与方差感知张力最大，最能区分策略）
    "both-anti":   dict(sigma_range=(0.2, 5.0),  cost_range=(1.0, 30.0), cost_sigma_corr=-0.6, hard=False),
    # 双强 + 困难小 gap
    "both-hard":   dict(sigma_range=(0.2, 5.0),  cost_range=(1.0, 30.0), cost_sigma_corr=0.3,  hard=True),
}

ALGOS_ADAPTIVE = {
    "confidence_greedy": confidence_greedy,
    "CACG": cacg,
    "CACG-noEB": cacg_no_eb,
    "CACG-noCost": cacg_no_cost,
    "CACG-MP": cacg_mp,
}


def run_regime(regime_name, params, seeds, n, universe, k, eta, delta, fixed_m):
    rows = {a: [] for a in ["fixed_guaranteed", *ALGOS_ADAPTIVE.keys(), "random"]}
    greedy_vals = []
    for seed in seeds:
        inst = make_instance(seed=seed, n=n, universe_size=universe, k=k,
                             name=f"{regime_name}_s{seed}", **params)
        g = exact_greedy(inst)
        greedy_vals.append(g.fval)
        thresh = (1 - 1/math.e) * g.fval - eta * k  # 可接受下界（对齐定理一）

        rng = np.random.default_rng(10_000 + seed)
        # 有保证的非自适应基线（worst-case m）
        orc = NoisyMarginalOracle(inst, rng)
        r = fixed_sampling_guaranteed(inst, orc, eta=eta, delta=delta)
        rows["fixed_guaranteed"].append(_row(r, g, thresh))
        # 自适应族
        for aname, fn in ALGOS_ADAPTIVE.items():
            orc = NoisyMarginalOracle(inst, rng)
            r = fn(inst, orc, eta=eta, delta=delta)
            rows[aname].append(_row(r, g, thresh))
        # 随机
        r = random_baseline(inst, rng)
        rows["random"].append(_row(r, g, thresh))
    return rows, float(np.mean(greedy_vals))


def _row(r, g, thresh):
    return dict(fval=r.fval, gap=g.fval - r.fval, queries=r.n_queries,
                cost=r.total_cost, cert=r.certificate,
                fail=int(r.fval < thresh - 1e-9),
                trunc=r.meta.get("n_truncated", 0))


def summarize(rows):
    out = {}
    for a, rs in rows.items():
        arr = {k: np.array([x[k] for x in rs], dtype=float) for k in rs[0]}
        out[a] = dict(
            f_mean=float(arr["fval"].mean()),
            gap_mean=float(arr["gap"].mean()),
            queries_mean=float(arr["queries"].mean()),
            cost_mean=float(arr["cost"].mean()),
            cost_std=float(arr["cost"].std()),
            cert_mean=float(arr["cert"].mean()),
            fail_rate=float(arr["fail"].mean()),
            trunc_mean=float(arr["trunc"].mean()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--universe", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--fixed_m", type=int, default=30)
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    seeds = list(range(6 if args.quick else 20))
    t0 = time.time()
    all_summary = {}
    for rname, params in REGIMES.items():
        rows, gmean = run_regime(rname, params, seeds, args.n, args.universe,
                                 args.k, args.eta, args.delta, args.fixed_m)
        s = summarize(rows)
        all_summary[rname] = dict(greedy_f_mean=gmean, algos=s)
        # 打印
        print(f"\n=== regime: {rname}  (greedy f≈{gmean:.1f}, seeds={len(seeds)}) ===")
        base = s["confidence_greedy"]["cost_mean"]
        fixed = s["fixed_guaranteed"]["cost_mean"]
        print(f"{'algo':18s} {'f':>6} {'gap':>5} {'queries':>9} {'cost':>11} "
              f"{'vsFixed':>8} {'vsConf':>7} {'fail':>5} {'trunc':>5}")
        for a in ["fixed_guaranteed", "confidence_greedy", "CACG-noEB", "CACG-noCost", "CACG-MP", "CACG", "random"]:
            d = s[a]
            vsf = f"{100*(1-d['cost_mean']/fixed):+5.0f}%" if fixed > 0 else "  n/a"
            vsc = f"{100*(1-d['cost_mean']/base):+5.0f}%" if base > 0 else "  n/a"
            print(f"{a:18s} {d['f_mean']:6.1f} {d['gap_mean']:5.1f} "
                  f"{d['queries_mean']:9.0f} {d['cost_mean']:11.1f} {vsf:>8} {vsc:>7} "
                  f"{d['fail_rate']:5.2f} {d.get('trunc_mean',0):5.1f}")

    import os
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), seeds=seeds, summary=all_summary), f, indent=2)
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
