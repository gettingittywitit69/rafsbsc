from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from statistics import NormalDist

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT

ANALYTIC_METHOD = "iid_normal_analytic"
GARCH_MLE_METHOD = "garch11_mle_analytic"

RESULT_COLUMNS = [
    "dgp",
    "method",
    "n",
    "S_true",
    "outer_reps",
    "coverage_95",
    "reject_rate_H0_S_eq_0",
    "se_cell",
    "mc_se",
    "mc_lo",
    "mc_hi",
]

DIAGNOSTIC_COLUMNS = [
    "dgp",
    "n",
    "S_true",
    "bias",
    "rmse",
    "mc_sd_S_hat",
    "outer_reps",
]

CI_SWEEP_COLUMNS = [
    "dgp",
    "method",
    "n",
    "S_true",
    "ci_level",
    "outer_reps",
    "coverage",
    "avg_ci_length",
    "mc_se",
    "mc_lo",
    "mc_hi",
]


@dataclass(frozen=True)
class SimConfig:
    """Minimal config matching the main simulation setup used in bsc.ipynb."""

    seed: int = 0
    alpha: float = 0.05
    R: int = 62025
    R_garch: int = 62025
    dgps: tuple[str, ...] = ("iid_normal", "garch11_t")
    methods: tuple[str, ...] = (ANALYTIC_METHOD, GARCH_MLE_METHOD)
    n_grid: tuple[int, ...] = (60, 400, 800, 1200, 1600)
    S_grid: tuple[float, ...] = (0.25, 0.5, 0.75)
    g_alpha: float = 0.05
    g_beta: float = 0.90
    nu: float = 7.0
    burn: int = 500
    ci_levels: tuple[float, ...] = (0.90, 0.95, 0.975, 0.99)


def stable_seed(*parts: object) -> int:
    # Stable cell-level seeds so each (dgp, n, S_true) is reproducible across runs.
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def simulate_iid_normal(rng: np.random.Generator, n: int, s_true: float, reps: int) -> np.ndarray:
    # Unit-vol normal returns with mean set equal to the true Sharpe.
    return float(s_true) + rng.standard_normal((int(reps), int(n)))


def simulate_garch11_t_stats(
    rng: np.random.Generator,
    n: int,
    s_true: float,
    reps: int,
    g_alpha: float,
    g_beta: float,
    nu: float,
    burn: int,
    store_reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Simulate many GARCH(1,1)-t paths in one pass and return:
    # 1) sample Sharpe estimates, 2) stored paths for per-rep MLE fitting.
    n_i = int(n)
    reps_i = int(reps)
    burn_i = int(burn)
    store_i = int(store_reps)
    total = n_i + burn_i

    mu = float(s_true)
    alpha = float(g_alpha)
    beta = float(g_beta)
    omega = 1.0 - alpha - beta
    nu_f = float(nu)
    scale = np.sqrt((nu_f - 2.0) / nu_f)

    # Start from unconditional variance 1 and evolve sigma_t^2 recursively.
    s2 = np.ones(reps_i, dtype=float)
    x_prev = np.empty(reps_i, dtype=float)
    sum_x = np.zeros(reps_i, dtype=float)
    sum_x2 = np.zeros(reps_i, dtype=float)
    x_fit = np.empty((store_i, n_i), dtype=float)

    for t in range(total):
        z_t = rng.standard_t(df=nu_f, size=reps_i) * scale
        if t > 0:
            resid_sq = (x_prev - mu) ** 2
            s2 = omega + alpha * resid_sq + beta * s2
        x_t = mu + np.sqrt(s2) * z_t
        if t >= burn_i:
            idx = t - burn_i
            sum_x += x_t
            sum_x2 += x_t**2
            x_fit[:, idx] = x_t[:store_i]
        x_prev = x_t

    mean = sum_x / float(n_i)
    var = (sum_x2 - (sum_x * sum_x) / float(n_i)) / float(n_i - 1)
    sd = np.sqrt(np.maximum(var, 1e-12))
    s_hat = mean / sd
    return s_hat, x_fit


def vectorized_sharpe(x: np.ndarray) -> np.ndarray:
    # Sharpe = sample mean / sample std, computed row-wise for all reps at once.
    return np.mean(x, axis=1) / np.maximum(np.std(x, axis=1, ddof=1), 1e-12)


def se_iid_analytic(s_hat: np.ndarray, n: int) -> np.ndarray:
    # Classical IID asymptotic SE for the Sharpe estimator.
    return np.sqrt((1.0 + 0.5 * s_hat**2) / float(n))


def h2_from_t_nu(nu: float) -> float:
    # Fourth-moment ratio for unit-variance Student-t innovations.
    nu_f = float(nu)
    return 3.0 * (nu_f - 2.0) / (nu_f - 4.0)


def omega_garch_closed_form(s_hat: float | np.ndarray, alpha1: float, beta: float, h2: float) -> float | np.ndarray:
    # Plug-in long-run variance factor for Sharpe under symmetric GARCH(1,1).
    alpha1_f = float(alpha1)
    beta_f = float(beta)
    h2_f = float(h2)
    gamma = alpha1_f + beta_f
    d = 1.0 - gamma**2 - (h2_f - 1.0) * alpha1_f**2
    denominator = (d + 1e-12) * ((1.0 - gamma) + 1e-12)
    factor = ((h2_f - 1.0) * (1.0 + gamma) * (1.0 - beta_f) ** 2) / denominator
    omega = 1.0 + (np.asarray(s_hat, dtype=float) ** 2 / 4.0) * factor
    return np.maximum(np.asarray(omega, dtype=float), 1e-12)


def fit_garch11_normal(x: np.ndarray):
    # One-step minimal MLE wrapper around arch.
    model = ConstantMean(np.asarray(x, dtype=float))
    model.volatility = GARCH(p=1, o=0, q=1)
    model.distribution = Normal()
    return model.fit(update_freq=0, disp="off", show_warning=False)


def fit_garch11_t(x: np.ndarray):
    # Same as above but with Student-t innovations.
    model = ConstantMean(np.asarray(x, dtype=float))
    model.volatility = GARCH(p=1, o=0, q=1)
    model.distribution = StudentsT()
    return model.fit(update_freq=0, disp="off", show_warning=False)


def coverage_with_mc_stats(cov: np.ndarray, outer_reps: int) -> tuple[float, float, float, float]:
    # Monte Carlo estimate + normal approximation error bars.
    p_hat = float(np.mean(np.asarray(cov, dtype=float)))
    mc_se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / float(outer_reps)))
    return p_hat, mc_se, float(p_hat - 1.96 * mc_se), float(p_hat + 1.96 * mc_se)


def build_ci_sweep_rows(
    dgp: str,
    method: str,
    n: int,
    s_true: float,
    ci_levels: tuple[float, ...],
    s_hat: np.ndarray,
    se: np.ndarray,
    outer_reps: int,
) -> list[dict[str, float | int | str]]:
    # Recompute CI coverage/length across requested confidence levels.
    rows: list[dict[str, float | int | str]] = []
    s_arr = np.asarray(s_hat, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    for ci_level in ci_levels:
        z = NormalDist().inv_cdf(0.5 + 0.5 * float(ci_level))
        half = z * se_arr
        lo = s_arr - half
        hi = s_arr + half
        cov = ((lo <= float(s_true)) & (float(s_true) <= hi)).astype(float)
        p_hat, mc_se, mc_lo, mc_hi = coverage_with_mc_stats(cov, outer_reps)
        rows.append(
            {
                "dgp": dgp,
                "method": method,
                "n": int(n),
                "S_true": float(s_true),
                "ci_level": float(ci_level),
                "outer_reps": int(outer_reps),
                "coverage": p_hat,
                "avg_ci_length": float(np.mean(2.0 * np.abs(half))),
                "mc_se": mc_se,
                "mc_lo": mc_lo,
                "mc_hi": mc_hi,
            }
        )
    return rows


def run_cell(
    dgp: str,
    n: int,
    s_true: float,
    cfg: SimConfig,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str], list[dict[str, float | int | str]]]:
    # One simulation "cell" = fixed (dgp, n, S_true), returning summary rows.
    rng = np.random.default_rng(stable_seed(cfg.seed, "cell", dgp, n, s_true))
    n_i = int(n)
    s_true_f = float(s_true)
    r_i = int(cfg.R)
    r_g = int(cfg.R_garch)

    if dgp == "iid_normal":
        # IID DGP: simulate raw paths and estimate Sharpe directly.
        x = simulate_iid_normal(rng, n_i, s_true_f, r_i)
        s_hat = vectorized_sharpe(x)
        x_for_fit = x[:r_g, :]
    else:
        # GARCH DGP: simulate and keep path samples for MLE plug-in step.
        s_hat, x_for_fit = simulate_garch11_t_stats(
            rng=rng,
            n=n_i,
            s_true=s_true_f,
            reps=r_i,
            g_alpha=cfg.g_alpha,
            g_beta=cfg.g_beta,
            nu=cfg.nu,
            burn=cfg.burn,
            store_reps=r_g,
        )

    z95 = NormalDist().inv_cdf(1.0 - float(cfg.alpha) / 2.0)
    result_rows: list[dict[str, float | int | str]] = []
    ci_rows: list[dict[str, float | int | str]] = []

    if ANALYTIC_METHOD in cfg.methods:
        # Method 1: IID analytic standard errors.
        se = se_iid_analytic(s_hat, n_i)
        lo = s_hat - z95 * se
        hi = s_hat + z95 * se
        cov = ((lo <= s_true_f) & (s_true_f <= hi)).astype(float)
        rej = ((lo > 0.0) | (hi < 0.0)).astype(float)
        p_hat, mc_se, mc_lo, mc_hi = coverage_with_mc_stats(cov, r_i)
        result_rows.append(
            {
                "dgp": dgp,
                "method": ANALYTIC_METHOD,
                "n": n_i,
                "S_true": s_true_f,
                "outer_reps": r_i,
                "coverage_95": p_hat,
                "reject_rate_H0_S_eq_0": float(np.mean(rej)),
                "se_cell": float(np.mean(se)),
                "mc_se": mc_se,
                "mc_lo": mc_lo,
                "mc_hi": mc_hi,
            }
        )
        ci_rows.extend(
            build_ci_sweep_rows(
                dgp=dgp,
                method=ANALYTIC_METHOD,
                n=n_i,
                s_true=s_true_f,
                ci_levels=tuple(cfg.ci_levels),
                s_hat=s_hat,
                se=se,
                outer_reps=r_i,
            )
        )

    if GARCH_MLE_METHOD in cfg.methods:
        # Method 2: per-rep GARCH MLE, then plug fitted params into closed-form omega.
        s_g = s_hat[:r_g]
        se_g = np.empty(r_g, dtype=float)
        cov_g = np.empty(r_g, dtype=float)
        rej_g = np.empty(r_g, dtype=float)
        # Keep model family aligned with the DGP used in this cell.
        use_normal_fit = dgp == "iid_normal"
        for i in range(r_g):
            res = fit_garch11_normal(x_for_fit[i, :]) if use_normal_fit else fit_garch11_t(x_for_fit[i, :])
            params = res.params
            alpha1 = float(params["alpha[1]"])
            beta = float(params["beta[1]"])
            h2 = 3.0 if use_normal_fit else h2_from_t_nu(float(params["nu"]))
            omega = float(omega_garch_closed_form(float(s_g[i]), alpha1, beta, h2))
            se = float(np.sqrt(omega / float(n_i)))
            lo = float(s_g[i] - z95 * se)
            hi = float(s_g[i] + z95 * se)
            se_g[i] = se
            cov_g[i] = float(lo <= s_true_f <= hi)
            rej_g[i] = float((lo > 0.0) or (hi < 0.0))

        p_hat, mc_se, mc_lo, mc_hi = coverage_with_mc_stats(cov_g, r_g)
        result_rows.append(
            {
                "dgp": dgp,
                "method": GARCH_MLE_METHOD,
                "n": n_i,
                "S_true": s_true_f,
                "outer_reps": r_g,
                "coverage_95": p_hat,
                "reject_rate_H0_S_eq_0": float(np.mean(rej_g)),
                "se_cell": float(np.mean(se_g)),
                "mc_se": mc_se,
                "mc_lo": mc_lo,
                "mc_hi": mc_hi,
            }
        )
        ci_rows.extend(
            build_ci_sweep_rows(
                dgp=dgp,
                method=GARCH_MLE_METHOD,
                n=n_i,
                s_true=s_true_f,
                ci_levels=tuple(cfg.ci_levels),
                s_hat=s_g,
                se=se_g,
                outer_reps=r_g,
            )
        )

    diag_row: dict[str, float | int | str] = {
        "dgp": dgp,
        "n": n_i,
        "S_true": s_true_f,
        "bias": float(np.mean(s_hat - s_true_f)),
        "rmse": float(np.sqrt(np.mean((s_hat - s_true_f) ** 2))),
        "mc_sd_S_hat": float(np.std(s_hat, ddof=1)),
        "outer_reps": r_i,
    }
    return result_rows, diag_row, ci_rows


def run_simulation(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Traverse the full experiment grid and collect row dictionaries first.
    result_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    ci_rows: list[dict[str, float | int | str]] = []

    for dgp, n, s_true in product(cfg.dgps, cfg.n_grid, cfg.S_grid):
        rows, diag, ci = run_cell(dgp, int(n), float(s_true), cfg)
        result_rows.extend(rows)
        diagnostic_rows.append(diag)
        ci_rows.extend(ci)

    # Build clean DataFrames in a deterministic sort order for easy table use.
    results = (
        pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
        .sort_values(["dgp", "n", "S_true", "method"])
        .reset_index(drop=True)
    )
    diagnostics = (
        pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
        .sort_values(["dgp", "n", "S_true"])
        .reset_index(drop=True)
    )
    ci_sweep = (
        pd.DataFrame(ci_rows, columns=CI_SWEEP_COLUMNS)
        .sort_values(["dgp", "n", "S_true", "method", "ci_level"])
        .reset_index(drop=True)
    )
    return results, diagnostics, ci_sweep


def main() -> None:
    # Default thesis run; writes plain CSV outputs in current working directory.
    cfg = SimConfig()
    results, diagnostics, ci_sweep = run_simulation(cfg)
    results.to_csv("results.csv", index=False)
    diagnostics.to_csv("diagnostics.csv", index=False)
    ci_sweep.to_csv("ci_sweep.csv", index=False)


if __name__ == "__main__":
    main()
