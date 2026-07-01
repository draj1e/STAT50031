"""
confidence.py
=============
置信半径构造（对应论文 §5.2 / §6.2）。

噪声设定：一次边际观测 X = Δ(e|S) + ξ，ξ 为尺度 σ_e 的次高斯（实现里用高斯）。
因此"经验均值 - 真值"的偏差尺度是 σ_e/√n，与边际大小无关。据此：

  - Hoeffding（方差-agnostic）：r = σ_bound · sqrt(2·ell/n)
        σ_bound 用一个"全局"上界（如所有臂的 σ_max）——不区分臂 -> 对低方差臂浪费。
  - Empirical-Bernstein（方差-adaptive，本文 CACG）：
        r = sqrt(2·var_hat·ell/n) + 3·dev·ell/n
        var_hat 为样本方差；dev 为"经验偏差界"= max_i|x_i-mean|（随 σ_e 自适应，且随 n 快速衰减）。
        低方差臂半径显著更小 -> 异方差场景省样本。

  - anytime / 时间一致修正：ell 里含 2·ln ln n，使"边采样边判停"合法。

约定：radius 给出 r 使得以概率≥1-δ 有 |mean - 真值| ≤ r（双侧）。
best-arm 内核对每臂用 δ_arm = δ/(n·k)。
"""
from __future__ import annotations
import numpy as np


def _ell(n_e: int, delta: float, anytime: bool, base_const: float) -> float:
    """对数因子：ln(base_const/δ) (+ 2 ln ln n 若 anytime)。"""
    n_e = max(int(n_e), 1)
    val = np.log(base_const / max(delta, 1e-300))
    if anytime:
        val += 2.0 * np.log(np.log(max(n_e, 3)))  # n≥3 保证 ln ln n>0
    return float(val)


def hoeffding_radius(n_e: int, sigma_bound: float, delta: float, anytime: bool = True) -> float:
    """次高斯 Hoeffding 双侧半径：r = sigma_bound · sqrt(2·ell/n)。"""
    if n_e <= 0:
        return np.inf
    ell = _ell(n_e, delta, anytime, base_const=2.0)   # 双侧 -> 常数 2
    return float(sigma_bound * np.sqrt(2.0 * ell / n_e))


def empirical_bernstein_radius(n_e: int, var_hat: float, sigma_hat: float, delta: float,
                               anytime: bool = True) -> float:
    """
    Maurer–Pontil (2009) 经验-Bernstein 双侧半径（严格，但小样本被 3R/n 项主导）：
        r = sqrt(2·var_hat·ell/n) + 3·sigma_hat·ell/n
    仅在 n 大、方差远小于范围时才优于 sub-Gaussian。保留作对照，不作 CACG 默认。
    """
    if n_e <= 1:
        return np.inf
    ell = _ell(n_e, delta, anytime, base_const=3.0)
    var_hat = max(var_hat, 0.0)
    sigma_hat = max(sigma_hat, 0.0)
    return float(np.sqrt(2.0 * var_hat * ell / n_e) + 3.0 * sigma_hat * ell / n_e)


def emp_subgaussian_radius(n_e: int, sigma_hat: float, delta: float,
                           anytime: bool = True, inflate: float = 1.0) -> float:
    """
    经验-方差 sub-Gaussian 半径（CACG 默认，异方差自适应的正确形式）：
        r = inflate · sigma_hat · sqrt(2·ell/n)
    用 **per-arm 经验标准差 sigma_hat** 替代全局 σ_max：低方差臂半径按其真实尺度收缩，
    且无 Maurer–Pontil 的 3R/n 重项，小样本即有效。
    噪声为 sub-Gaussian 时理论上应以真 σ 计；这里用 σ̂（经验），以 inflate(≥1) 补偿小样本低估，
    覆盖率由实验经验失败率校验（对齐定理二的高概率事件）。
    """
    if n_e <= 1:
        return np.inf
    ell = _ell(n_e, delta, anytime, base_const=2.0)  # 双侧
    return float(inflate * sigma_hat * np.sqrt(2.0 * ell / n_e))


class ArmStats:
    """单个候选(臂)的在线统计：均值、样本方差(Welford)、样本数、经验偏差界。"""
    __slots__ = ("n", "mean", "M2", "_min", "_max")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self._min = np.inf
        self._max = -np.inf

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    def radius(self, method: str, delta: float, sigma_bound: float,
               anytime: bool = True, inflate: float = 1.0) -> float:
        if self.n <= 0:
            return np.inf
        if method == "hoeffding":
            return hoeffding_radius(self.n, sigma_bound, delta, anytime)
        elif method == "emp-subg":
            return emp_subgaussian_radius(self.n, np.sqrt(max(self.var, 0.0)),
                                          delta, anytime, inflate)
        elif method == "emp-bernstein":
            v = self.var
            return empirical_bernstein_radius(self.n, v, np.sqrt(max(v, 0.0)), delta, anytime)
        else:
            raise ValueError(f"unknown method {method}")


if __name__ == "__main__":
    # 覆盖率自检：对不同真实 σ，非自适应半径的覆盖率应 ≥ 1-δ。
    rng = np.random.default_rng(0)
    delta = 0.1
    true_mu = 3.0
    trials = 4000
    print(f"target coverage ≥ {1-delta}")
    for sigma in [0.5, 2.0, 6.0]:
        for method, sbound in [("hoeffding", 6.0), ("emp-bernstein", None)]:
            for m in [10, 50, 200]:
                cov = 0
                for _ in range(trials):
                    xs = true_mu + rng.normal(0, sigma, size=m)
                    st = ArmStats()
                    for x in xs:
                        st.update(x)
                    r = st.radius(method, delta, sigma_bound=(sbound or 6.0), anytime=False)
                    if abs(st.mean - true_mu) <= r:
                        cov += 1
                tag = f"{method}(σb={sbound})" if method == "hoeffding" else method
                print(f"  true σ={sigma:>3} {tag:22s} m={m:4d}  coverage={cov/trials:.3f}")
