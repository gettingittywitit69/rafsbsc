"""Importable runtime helpers for ``notebooks/bsc_final.ipynb``.

Keeping the worker entrypoint in a real module allows ``ProcessPoolExecutor``
to use spawned processes from a notebook environment.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

__all__ = [
    "Config",
    "run_cell",
    "run_partA",
]


def sharpe_hat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return np.nan if (not np.isfinite(s) or s <= 0) else x.mean() / s


def wald_ci(S_hat: float, se: float, alpha: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return (S_hat - z * se, S_hat + z * se)


def simulate_iid_normal(
    rng: np.random.Generator, n: int, S_true: float, sigma: float = 1.0
) -> np.ndarray:
    mu = S_true * sigma
    return mu + sigma * rng.standard_normal(n)


def simulate_garch11_t(
    rng: np.random.Generator,
    n: int,
    S_true: float,
    g_alpha: float,
    g_beta: float,
    nu: float,
    sigma_uncond: float = 1.0,
    burn: int = 500,
) -> np.ndarray:
    if g_alpha + g_beta >= 1:
        raise ValueError("Need g_alpha + g_beta < 1 for finite unconditional variance.")
    if nu <= 2:
        raise ValueError("Need nu > 2 to standardize t innovations to Var=1.")

    omega = (1.0 - g_alpha - g_beta) * (sigma_uncond**2)
    mu = S_true * sigma_uncond

    m = burn + n
    h = np.empty(m, float)
    eps = np.empty(m, float)
    x = np.empty(m, float)

    z = rng.standard_t(df=nu, size=m) * np.sqrt((nu - 2.0) / nu)

    h[0] = sigma_uncond**2
    eps[0] = np.sqrt(h[0]) * z[0]
    x[0] = mu + eps[0]

    for t in range(1, m):
        h[t] = omega + g_alpha * (eps[t - 1] ** 2) + g_beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * z[t]
        x[t] = mu + eps[t]

    return x[burn:]


def simulate_from_true_dgp(
    rng: np.random.Generator, dgp: str, n: int, S_true: float, cfg: "Config"
) -> np.ndarray:
    if dgp == "iid_normal":
        return simulate_iid_normal(rng, n=n, S_true=S_true, sigma=1.0)
    if dgp == "garch11_t":
        return simulate_garch11_t(
            rng,
            n=n,
            S_true=S_true,
            g_alpha=cfg.g_alpha,
            g_beta=cfg.g_beta,
            nu=cfg.nu,
            sigma_uncond=1.0,
            burn=cfg.burn,
        )
    raise ValueError(f"Unknown dgp: {dgp}")


def fit_model_to_data(dgp: str, x: np.ndarray, cfg: "Config") -> dict:
    x = np.asarray(x, float)
    mu_hat = float(np.nanmean(x))
    sigma_hat = float(np.nanstd(x, ddof=1))
    if not np.isfinite(sigma_hat) or sigma_hat <= 0:
        sigma_hat = np.nan

    if dgp == "iid_normal":
        return {"model": "iid_normal", "mu": mu_hat, "sigma": sigma_hat}

    if dgp == "garch11_t":
        return {
            "model": "garch11_t",
            "mu": mu_hat,
            "sigma_uncond": sigma_hat,
            "g_alpha": cfg.g_alpha,
            "g_beta": cfg.g_beta,
            "nu": cfg.nu,
            "burn": cfg.burn,
        }

    raise ValueError(f"Unknown model for fit: {dgp}")


def simulate_from_fitted_model(
    rng: np.random.Generator, theta_hat: dict, n: int
) -> np.ndarray:
    model = theta_hat["model"]

    if model == "iid_normal":
        mu = float(theta_hat["mu"])
        sigma = float(theta_hat["sigma"])
        if not np.isfinite(sigma) or sigma <= 0:
            return np.full(n, np.nan)
        return mu + sigma * rng.standard_normal(n)

    if model == "garch11_t":
        mu = float(theta_hat["mu"])
        sigma_uncond = float(theta_hat["sigma_uncond"])
        if not np.isfinite(sigma_uncond) or sigma_uncond <= 0:
            return np.full(n, np.nan)

        s_impl = mu / sigma_uncond
        return simulate_garch11_t(
            rng,
            n=n,
            S_true=s_impl,
            g_alpha=float(theta_hat["g_alpha"]),
            g_beta=float(theta_hat["g_beta"]),
            nu=float(theta_hat["nu"]),
            sigma_uncond=sigma_uncond,
            burn=int(theta_hat["burn"]),
        )

    raise ValueError(f"Unknown fitted model: {model}")


def se_cell_parametric_bootstrap_precompute(
    dgp: str,
    n: int,
    S_true: float,
    cfg: "Config",
    rng: np.random.Generator,
) -> float:
    x_ref = simulate_from_true_dgp(rng, dgp=dgp, n=n, S_true=S_true, cfg=cfg)
    theta_hat = fit_model_to_data(dgp, x_ref, cfg)

    b = max(2, int(cfg.B_cell))
    s_star = np.empty(b, float)
    for idx in range(b):
        x_star = simulate_from_fitted_model(rng, theta_hat=theta_hat, n=n)
        s_star[idx] = sharpe_hat(x_star)

    return float(np.nanstd(s_star, ddof=1))


def se_iid_analytic(S_hat: float, n: int) -> float:
    return float(np.sqrt((1.0 + 0.5 * S_hat**2) / n))


@dataclass(frozen=True)
class Config:
    seed: int = 0
    alpha: float = 0.05
    R: int = 300
    B_cell: int = 300
    dgps: tuple[str, ...] = ("iid_normal", "garch11_t")
    methods: tuple[str, ...] = ("iid_normal_analytic", "cell_parametric_bootstrap_wald")
    n_grid: tuple[int, ...] = (36, 60, 120)
    S_grid: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    g_alpha: float = 0.05
    g_beta: float = 0.90
    nu: float = 7.0
    burn: int = 500
    max_workers: int = max(1, (os.cpu_count() or 2) - 1)


def stable_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def run_cell(dgp: str, n: int, S_true: float, cfg_dict: dict) -> tuple[list[dict], dict]:
    cfg = Config(**cfg_dict)
    rng_pre = np.random.default_rng(stable_seed(cfg.seed, "pre", dgp, n, S_true))
    rng_rep = np.random.default_rng(stable_seed(cfg.seed, "rep", dgp, n, S_true))

    se_cell = se_cell_parametric_bootstrap_precompute(
        dgp=dgp, n=n, S_true=S_true, cfg=cfg, rng=rng_pre
    )
    omega_hat_cell = float(n * (se_cell**2)) if np.isfinite(se_cell) else np.nan

    s_hats: list[float] = []
    se_a: list[float] = []
    cov_a: list[float] = []
    rej_a: list[float] = []
    se_b: list[float] = []
    cov_b: list[float] = []
    rej_b: list[float] = []

    for _ in range(cfg.R):
        x = simulate_from_true_dgp(rng_rep, dgp=dgp, n=n, S_true=S_true, cfg=cfg)
        s_hat = sharpe_hat(x)
        s_hats.append(s_hat)

        se = se_iid_analytic(s_hat, n)
        lo, hi = wald_ci(s_hat, se, cfg.alpha)
        se_a.append(se)
        cov_a.append(float(lo <= S_true <= hi))
        rej_a.append(float((lo > 0.0) or (hi < 0.0)))

        se = se_cell
        lo, hi = wald_ci(s_hat, se, cfg.alpha)
        se_b.append(se)
        cov_b.append(float(lo <= S_true <= hi))
        rej_b.append(float((lo > 0.0) or (hi < 0.0)))

    s_arr = np.asarray(s_hats, float)
    mc_sd = float(np.nanstd(s_arr, ddof=1))
    bias = float(np.nanmean(s_arr - S_true))
    rmse = float(np.sqrt(np.nanmean((s_arr - S_true) ** 2)))

    diagnostics_row = {
        "dgp": dgp,
        "n": int(n),
        "S_true": float(S_true),
        "bias": bias,
        "rmse": rmse,
        "mc_sd_S_hat": mc_sd,
    }

    def pack(method: str, se_list, cov_list, rej_list) -> dict:
        se_arr = np.asarray(se_list, float)
        cov_arr = np.asarray(cov_list, float)
        rej_arr = np.asarray(rej_list, float)
        return {
            "dgp": dgp,
            "n": int(n),
            "S_true": float(S_true),
            "method": method,
            "coverage_95": float(np.nanmean(cov_arr)),
            "reject_rate_H0_S_eq_0": float(np.nanmean(rej_arr)),
            "se_ratio_meanSE_over_mcSD": float(np.nanmean(se_arr) / mc_sd)
            if mc_sd > 0
            else np.nan,
            "fit_fail_rate": float(np.mean(np.isnan(se_arr))),
            "se_cell": float(se_cell) if method == "cell_parametric_bootstrap_wald" else np.nan,
            "omega_hat_cell": float(omega_hat_cell)
            if method == "cell_parametric_bootstrap_wald"
            else np.nan,
        }

    method_rows = [
        pack("iid_normal_analytic", se_a, cov_a, rej_a),
        pack("cell_parametric_bootstrap_wald", se_b, cov_b, rej_b),
    ]
    return method_rows, diagnostics_row


def run_partA(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(dgp, n, s) for dgp in cfg.dgps for n in cfg.n_grid for s in cfg.S_grid]
    cfg_dict = asdict(cfg)

    method_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    if cfg.max_workers <= 1:
        for (dgp, n, s) in tasks:
            rows_m, row_d = run_cell(dgp, n, s, cfg_dict)
            method_rows.extend(rows_m)
            diagnostics_rows.append(row_d)
    else:
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(run_cell, dgp, n, s, cfg_dict) for (dgp, n, s) in tasks]
            for fut in as_completed(futs):
                rows_m, row_d = fut.result()
                method_rows.extend(rows_m)
                diagnostics_rows.append(row_d)

    df_methods = pd.DataFrame(method_rows).sort_values(
        ["dgp", "n", "S_true", "method"]
    ).reset_index(drop=True)
    df_diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["dgp", "n", "S_true"]
    ).reset_index(drop=True)
    return df_methods, df_diagnostics
