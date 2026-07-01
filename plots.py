"""
plots.py
========
由 results/summary_final.json 生成论文 §7 图（对应施工方案 II.4 指标可视化）。
产出：
  results/fig_cost_by_regime.png   各 regime 下总查询成本对比（对数刻度，突出 CACG 优势）
  results/fig_vs_conf.png          CACG 及消融相对 confidence_greedy 的成本降低百分比
另：budget.py 生成 anytime 预算-质量曲线（见其 __main__）。
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")


def load(path=None):
    path = path or os.path.join(RES, "summary_final.json")
    with open(path) as f:
        return json.load(f)


def fig_cost_by_regime(data):
    regimes = list(data["summary"].keys())
    algos = ["fixed_guaranteed", "confidence_greedy", "CACG-noEB", "CACG-noCost", "CACG"]
    labels = ["Fixed(guaranteed)", "Confidence-Greedy", "CACG-noVar", "CACG-noCost", "CACG(full)"]
    colors = ["#888", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    x = np.arange(len(regimes))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (a, lab, c) in enumerate(zip(algos, labels, colors)):
        vals = [data["summary"][r]["algos"][a]["cost_mean"] for r in regimes]
        ax.bar(x + (i - 2) * w, vals, w, label=lab, color=c)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=10)
    ax.set_ylabel("mean total query cost (log scale)")
    ax.set_title("Total query cost across noise/cost regimes (lower is better)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    p = os.path.join(RES, "fig_cost_by_regime.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("saved", p)


def fig_vs_conf(data):
    regimes = list(data["summary"].keys())
    algos = ["CACG-noEB", "CACG-noCost", "CACG"]
    labels = ["CACG-noVar (cost-aware only)", "CACG-noCost (variance-aware only)", "CACG (full)"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    x = np.arange(len(regimes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (a, lab, c) in enumerate(zip(algos, labels, colors)):
        red = []
        for r in regimes:
            base = data["summary"][r]["algos"]["confidence_greedy"]["cost_mean"]
            cur = data["summary"][r]["algos"][a]["cost_mean"]
            red.append(100 * (1 - cur / base))
        ax.bar(x + (i - 1) * w, red, w, label=lab, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=10)
    ax.set_ylabel("total-cost reduction vs Confidence-Greedy (%)")
    ax.set_title("Ablation: variance-aware vs cost-aware contribution (higher is better)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(RES, "fig_vs_conf.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    data = load()
    fig_cost_by_regime(data)
    fig_vs_conf(data)
