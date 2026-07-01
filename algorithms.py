"""
algorithms.py
=============
子模最大化算法族（对应论文 §3.2, §5, §6）：

  - exact_greedy            精确贪心（参照上界，无噪声）
  - fixed_sampling_greedy   固定采样贪心（每候选每轮采 m 次，取经验均值最大）
  - confidence_greedy       置信度贪心（淘汰式 BAI；成本无关、方差无关：Hoeffding + most-uncertain 采样）
  - cacg                    本文 CACG（淘汰式 BAI + 经验-Bernstein[异方差] + 成本感知采样 r_e/√c_e）
  - random_baseline         随机基线

统一通过一个 BAI 内核 `_bai_pick_round` 实现，参数化：
    method   ∈ {"hoeffding","emp-bernstein"}      置信半径（是否方差自适应）
    sampling ∈ {"most-uncertain","cost-aware"}     采样顺序（是否成本感知）
于是可做干净的 2×2 消融：EB on/off × cost-aware on/off。

返回统一的 Result：S、f(S)、查询次数、总成本、证书 L、每轮所选元素。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from submodular import CoverageInstance, NoisyMarginalOracle, true_marginal_range
from confidence import ArmStats

# 成本感知采样规则的 n 指数：score = σ̂/(c·n^p)。理论最优 p=1.5（见 §6.3 推导）。
# 由 budget_and_convergence.cost_exp_scan 扫描以验证收敛(C4)。
_COST_EXP = 1.5


@dataclass
class Result:
    name: str
    S: list
    fval: int
    n_queries: int
    total_cost: float
    certificate: float           # L = Σ LCB(所选边际)，f(S) 的高概率下界
    picks: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ----------------------------- 参照算法 -----------------------------
def exact_greedy(inst: CoverageInstance) -> Result:
    """精确贪心：每轮选真实最大边际。近似比≥1-1/e。查询次数记为 n·k（精确边际）。"""
    S, covered = [], 0
    for _ in range(inst.k):
        margs = inst.marginal_all_exact(covered)
        margs[S] = -1  # 已选不再选
        e = int(np.argmax(margs))
        if margs[e] <= 0:
            break
        S.append(e)
        covered |= inst._mask[e]
    return Result("exact_greedy", S, inst.f(S), inst.n * inst.k, 0.0,
                  certificate=float(inst.f(S)), picks=list(S))


def random_baseline(inst: CoverageInstance, rng: np.random.Generator) -> Result:
    S = list(rng.choice(inst.n, size=min(inst.k, inst.n), replace=False))
    return Result("random", S, inst.f(S), 0, 0.0, certificate=0.0, picks=list(S))


# ----------------------------- 固定采样贪心 -----------------------------
def fixed_sampling_greedy(inst: CoverageInstance, oracle: NoisyMarginalOracle,
                          m: int = 30) -> Result:
    """每轮对每个候选采样 m 次，取经验均值最大者。无自适应 -> 成本高。"""
    S, covered = [], 0
    cert = 0.0
    for _ in range(inst.k):
        remaining = [e for e in range(inst.n) if e not in S]
        best_e, best_mean = None, -math.inf
        for e in remaining:
            st = ArmStats()
            for _ in range(m):
                st.update(oracle.query(e, covered))
            if st.mean > best_mean:
                best_mean, best_e = st.mean, e
        if best_e is None:
            break
        S.append(best_e)
        covered |= inst._mask[best_e]
        cert += best_mean  # 固定采样无置信下界，用经验均值近似证书
    return Result(f"fixed_sampling(m={m})", S, inst.f(S),
                  oracle.n_queries, oracle.total_cost, certificate=cert, picks=list(S),
                  meta=dict(n_truncated=0))


def fixed_worstcase_m(inst, eta: float, delta: float) -> int:
    """非自适应贪心为保证 η-最优选择(w.p.≥1-δ)所需的每臂样本数(Hoeffding, 全局 σ_max)。
       每臂均值估计到 η/2 即可 -> m ≥ 8 σ_max² ln(2nk/δ) / η²。这是"无自适应"的诚实代价。"""
    sig = float(inst.sigma.max())
    return int(math.ceil(8.0 * sig * sig * math.log(2 * inst.n * inst.k / delta) / (eta * eta)))


def fixed_sampling_guaranteed(inst, oracle, *, eta=0.5, delta=0.1, **_) -> Result:
    """有保证的非自适应基线：m 取 worst-case 值。体现'不自适应=处处付最坏代价'。"""
    m = fixed_worstcase_m(inst, eta, delta)
    r = fixed_sampling_greedy(inst, oracle, m=m)
    r.name = f"fixed_guaranteed(m={m})"
    return r


# ----------------------------- BAI 内核（淘汰式） -----------------------------
def _bai_pick_round(inst, oracle, covered, active, *, method, cost_aware,
                    eta, delta_arm, sigma_ub, anytime, inflate=1.0,
                    init_pulls=3, max_pulls=8000, budget_left=math.inf, batch=4):
    """
    LUCB 型 (η,δ)-PAC top-1 identification（Kalyanakrishnan et al. 2012 思路）。
    每次只采样两个"决定性"臂——领先者 leader 与最强挑战者 challenger——而非全局最不确定臂，
    因此样本量随实例相关的 gap 自适应，不会把预算浪费在明显更差的高方差臂上。
      - leader     = argmax LCB
      - challenger = argmax_{e≠leader} UCB（在存活集内）
      - 淘汰：UCB_e < LCB_leader 的臂剔除
      - 停止：LCB_leader ≥ UCB_challenger − η（领先者 η-最优）；或仅剩一臂；或预算/上限
      - 采样对象：在 {leader, challenger} 中，成本无关时选半径大者；成本感知时选 r_e/√c_e 大者
        （square-root 规则，源自 Cost-Aware BAI）。
    返回 (leader, LCB_leader, cost_used)。
    """
    oracle.new_round(covered)
    stats = {e: ArmStats() for e in active}
    cost0 = oracle.total_cost
    for e in active:
        for _ in range(init_pulls):
            stats[e].update(oracle.query(e, covered))
    rad = {e: stats[e].radius(method, delta_arm, sigma_ub, anytime, inflate) for e in active}
    active_set = set(active)
    pulls = init_pulls * len(active)

    def resample(e, b):
        nonlocal pulls
        for _ in range(b):
            stats[e].update(oracle.query(e, covered))
        pulls += b
        rad[e] = stats[e].radius(method, delta_arm, sigma_ub, anytime, inflate)

    leader = active[0]
    truncated = False
    while True:
        leader = max(active_set, key=lambda e: stats[e].mean - rad[e])
        lcb_lead = stats[leader].mean - rad[leader]
        # 淘汰被支配者
        active_set = {e for e in active_set
                      if stats[e].mean + rad[e] >= lcb_lead - 1e-12}
        active_set.add(leader)
        if len(active_set) == 1:
            break
        challenger = max((e for e in active_set if e != leader),
                         key=lambda e: stats[e].mean + rad[e])
        ucb_ch = stats[challenger].mean + rad[challenger]
        # η-最优停止
        if lcb_lead >= ucb_ch - eta:
            break
        if (oracle.total_cost - cost0) >= budget_left or pulls >= max_pulls:
            truncated = True
            break
        # 在两个决定性臂中挑采样对象
        if cost_aware:
            # 停止判据依赖两臂半径之和 r_l+r_c，r_e=σ_e√(2ℓ/n_e)。对臂 e 多采一次使 r_e 下降
            # ∝ σ_e·n_e^{-3/2}，成本 c_e，故"单位成本收缩量" = σ̂_e/(c_e·n_e^{3/2})。
            # 贪心地采该值最大的臂 —— 等价于把预算按最优分配 n_e ∝ (σ_e/c_e)^{2/3} 铺开。
            def _score(e):
                sig = math.sqrt(max(stats[e].var, 1e-12))
                ne = max(stats[e].n, 1)
                return sig / (inst.cost[e] * ne ** _COST_EXP)
            sl, sc = _score(leader), _score(challenger)
        else:
            sl, sc = rad[leader], rad[challenger]   # 成本无关：采更不确定者（半径大者）
        target = leader if sl >= sc else challenger
        b = max(1, min(batch, max_pulls - pulls))
        if budget_left < math.inf:
            room = int((budget_left - (oracle.total_cost - cost0)) / max(inst.cost[target], 1e-9))
            b = max(1, min(b, room)) if room > 0 else 1
        resample(target, b)

    r_lead = stats[leader].radius(method, delta_arm, sigma_ub, anytime, inflate)
    return leader, stats[leader].mean - r_lead, oracle.total_cost - cost0, truncated


def bai_greedy(inst: CoverageInstance, oracle: NoisyMarginalOracle, *,
               name: str, method: str, cost_aware: bool,
               eta: float = 0.5, delta: float = 0.1, anytime: bool = True,
               inflate: float = 1.0, budget: float = math.inf) -> Result:
    """淘汰式置信度贪心的通用外壳。confidence_greedy / CACG / 消融变体都是它的实例。"""
    sigma_ub = float(inst.sigma.max())          # 已知的"全局"噪声上界（但非 per-arm）
    delta_arm = delta / max(inst.n * inst.k, 1)
    S, covered, cert = [], 0, 0.0
    spent = 0.0
    n_trunc = 0
    for _ in range(inst.k):
        active = [e for e in range(inst.n) if e not in S]
        if not active:
            break
        leader, lcb_lead, cost_used, truncated = _bai_pick_round(
            inst, oracle, covered, active, method=method, cost_aware=cost_aware,
            eta=eta, delta_arm=delta_arm, sigma_ub=sigma_ub, anytime=anytime,
            inflate=inflate, budget_left=budget - spent)
        spent += cost_used
        n_trunc += int(truncated)
        S.append(leader)
        covered |= inst._mask[leader]
        cert += lcb_lead
        if budget - spent <= 0:
            break
    return Result(name, S, inst.f(S), oracle.n_queries, oracle.total_cost,
                  certificate=cert, picks=list(S),
                  meta=dict(method=method, cost_aware=cost_aware, eta=eta, delta=delta,
                            n_truncated=n_trunc))


# ----------------------------- 具体算法 = BAI 内核的配置 -----------------------------
def confidence_greedy(inst, oracle, **kw) -> Result:
    """成本无关 + 方差无关基线：Hoeffding(全局 σ_max) + 成本无关采样。"""
    return bai_greedy(inst, oracle, name="confidence_greedy",
                      method="hoeffding", cost_aware=False, **kw)


def cacg(inst, oracle, **kw) -> Result:
    """本文 CACG（完整）：经验方差 sub-Gaussian[异方差自适应] + 成本感知采样。"""
    return bai_greedy(inst, oracle, name="CACG",
                      method="emp-subg", cost_aware=True, **kw)


def cacg_no_eb(inst, oracle, **kw) -> Result:
    """消融：去掉异方差自适应（用全局 σ_max Hoeffding），保留成本感知。"""
    return bai_greedy(inst, oracle, name="CACG-noEB",
                      method="hoeffding", cost_aware=True, **kw)


def cacg_no_cost(inst, oracle, **kw) -> Result:
    """消融：去掉成本感知，保留异方差自适应。"""
    return bai_greedy(inst, oracle, name="CACG-noCost",
                      method="emp-subg", cost_aware=False, **kw)


def cacg_mp(inst, oracle, **kw) -> Result:
    """对照：用 Maurer–Pontil 经验-Bernstein 半径的成本感知版（验证小样本被 3R/n 主导）。"""
    return bai_greedy(inst, oracle, name="CACG-MP",
                      method="emp-bernstein", cost_aware=True, **kw)


if __name__ == "__main__":
    from benchmark import make_instance, optimal_bruteforce
    inst = make_instance(seed=1, n=14, universe_size=80, k=4)
    opt = optimal_bruteforce(inst)
    g = exact_greedy(inst)
    print(f"OPT={opt}  exact_greedy f={g.fval}  ratio={g.fval/opt:.3f}  picks={g.picks}")
    rng = np.random.default_rng(1)
    for algo in [fixed_sampling_greedy, confidence_greedy, cacg, cacg_no_eb, cacg_no_cost]:
        oracle = NoisyMarginalOracle(inst, rng)
        r = algo(inst, oracle) if algo is fixed_sampling_greedy else algo(inst, oracle)
        print(f"{r.name:22s} f={r.fval:3d}  gap2greedy={g.fval-r.fval:+d}  "
              f"queries={r.n_queries:6d}  cost={r.total_cost:9.1f}  cert={r.certificate:6.1f}")
