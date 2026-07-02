# Submodular Maximization under Noisy Value Queries — Cost-Aware Confidence Greedy (CACG)

Reference implementation and experiments for **CACG**, a cost-aware, variance-adaptive
confidence-elimination greedy algorithm for monotone submodular maximization under a
cardinality constraint when marginal-gain queries are **noisy** (re-sampleable, independent,
sub-Gaussian) with **heteroscedastic noise** and **heterogeneous per-query costs**.

The algorithm treats each greedy round as a cost-aware best-arm identification (LUCB) problem
and allocates the sampling budget to minimize **total query cost** (not query count) while
retaining a high-probability $(1-1/e)\,\mathrm{OPT}-\varepsilon$ guarantee.

> Course project for STAT50031 (Introduction to Algorithms). This repository accompanies the
> paper and contains everything needed to reproduce the empirical results.

## Paper

The full survey paper (LaTeX source + compiled PDF) is in [`paper/`](paper/):
[`paper/main.pdf`](paper/main.pdf). To rebuild it (TeX Live; no external algorithm package needed):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 recommended
pip install -r requirements.txt                        # numpy + matplotlib
```

## Reproduce

```bash
python run_experiments.py --eta 0.75 --out results/summary_final.json   # main study (20 seeds × 5 regimes, ~5 min)
python budget_and_convergence.py                                        # anytime budget curve + cost-exponent convergence (C4)
python plots.py                                                         # regenerate the 3 figures
```

Quick pass: `python run_experiments.py --quick` (6 seeds). Every module also has a `__main__`
self-check and can be run standalone (`submodular.py`, `benchmark.py`, `confidence.py`,
`algorithms.py`).

## Files

| File | Role |
|---|---|
| `submodular.py` | Max-coverage objective (bitmask + `bit_count`); noisy marginal oracle (heteroscedastic $\sigma_e$, heterogeneous cost $c_e$, per-round caching of true marginals) |
| `benchmark.py` | Synthetic instance generator (controllable $\sigma$/$c$ spread, $\sigma$–$c$ correlation, hard small-gap instances) |
| `confidence.py` | Confidence radii: Hoeffding / empirical-variance sub-Gaussian (CACG default) / Maurer–Pontil empirical-Bernstein; anytime ($\ln\ln n$) correction; `ArmStats` online mean/variance |
| `algorithms.py` | Exact greedy, guaranteed fixed-sampling, **LUCB kernel** `_bai_pick_round`, `confidence_greedy` / `cacg` / ablations; cost exponent `_COST_EXP` |
| `run_experiments.py` | Main driver: 5 regimes × 20 seeds; aggregates quality / query count / total cost / failure rate / truncations |
| `budget_and_convergence.py` | Anytime budget-quality curve + certificate validity; cost-exponent $p$ scan (C4 convergence check) |
| `plots.py` | `fig_cost_by_regime.png`, `fig_vs_conf.png` |
| `results/` | Committed outputs: `summary_final.json`, `supp.json`, and the three figures |

## Key design decisions (each arrived at by empirical iteration)

1. **LUCB, not full-elimination round-robin.** Each step samples only the two decisive arms
   (leader + challenger), so the sample count adapts to the instance gap instead of wasting
   budget on clearly-worse high-variance arms.
2. **Empirical-variance sub-Gaussian radius** ($r_e=\hat\sigma_e\sqrt{2\ell/n_e}$): per-arm
   empirical $\sigma$ gives heteroscedastic adaptivity. Beats Maurer–Pontil, whose $3R\ell/n$
   term dominates at small samples.
3. **Cost-aware allocation** $\hat\sigma_e/(c_e\,n_e^{3/2})$: derived from "the stopping rule
   depends on the sum of the two arms' radii; pull the arm with the largest per-cost shrink",
   equivalent to the optimal allocation $n_e\propto(\sigma_e/c_e)^{2/3}$.

## Main results (20 seeds, η=0.75, δ=0.1; total query cost, lower is better)

| regime | fixed_guaranteed | confidence_greedy | CACG | vs fixed | vs conf |
|---|--:|--:|--:|--:|--:|
| mild | 539198 | 18948 | 10102 | −98% | +47% |
| hetero-var | 5733512 | 117324 | 27575 | −100% | +76% |
| hetero-cost | 2535830 | 88494 | 43501 | −98% | +51% |
| both-anti | 26922985 | 564666 | 77960 | −100% | +86% |
| both-hard | 26851407 | 418125 | 114570 | −98% | +73% |

- **Correctness (C1):** empirical failure rate = 0.00 in all regimes ($f(S)\ge(1-1/e)\mathrm{OPT}-\varepsilon$ holds).
- **Efficiency (C2):** CACG cuts total cost by 98–100% vs the guaranteed non-adaptive baseline and by 47–86% vs certifying confidence-greedy.
- **Attribution (C3):** variance-adaptivity is the dominant lever (CACG-noCost alone: +48–86%); cost-awareness adds on top in cost-dominated / hard regimes (both-hard +62%→+73%).
- **Convergence (C4):** total cost varies only ~5% for the cost exponent $p\in[1,2]$ (default 1.5 vs empirical-best 1.25 differ by 2.3%).
- **Anytime + certificate (C5):** budget-quality curve is monotone and plateaus at greedy; certificate validity $\Pr[L\le f(S)]=0.976\approx 1-\delta$.

## Honest scope

Gains hold under **benign re-sampleable sub-Gaussian noise with a direct marginal oracle**.
The individual ingredients (variance adaptivity, cost-aware allocation, anytime certificates)
each have precedents; the contribution here is combining them, with **total query cost** as
the objective, inside the submodular greedy loop — not a new principle. Persistent/adversarial
noise and the shared-baseline correlation that arises when only $f$ (not marginals) is
queryable are out of scope (discussed as future work in the paper).
