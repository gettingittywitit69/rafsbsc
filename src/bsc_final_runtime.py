"""Importable runtime helpers for ``notebooks/bsc_final.ipynb``.

Keeping the worker entrypoint in a real module allows ``ProcessPoolExecutor``
to use spawned processes from a notebook environment.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from . import sharpe_mc

__all__ = [
    "ANALYTIC_METHOD",
    "BOOTSTRAP_METHOD",
    "Config",
    "fit_model_to_data",
    "run_cell",
    "run_partA",
    "se_iid_analytic",
    "se_rep_parametric_bootstrap",
    "sharpe_hat",
    "simulate_from_fitted_model",
    "simulate_from_true_dgp",
    "simulate_garch11_t",
    "simulate_iid_normal",
    "stable_seed",
]


ANALYTIC_METHOD = "iid_normal_analytic"
BOOTSTRAP_METHOD = "cell_parametric_bootstrap_wald"
SUPPORTED_DGPS = ("iid_normal", "garch11_t")
SUPPORTED_METHODS = (ANALYTIC_METHOD, BOOTSTRAP_METHOD)


def sharpe_hat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return np.nan if (not np.isfinite(s) or s <= 0) else float(x.mean() / s)


def wald_ci(S_hat: float, se: float, alpha: float) -> tuple[float, float]:
    if not np.isfinite(S_hat) or not np.isfinite(se) or se <= 0:
        return (np.nan, np.nan)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return (float(S_hat - z * se), float(S_hat + z * se))


def simulate_iid_normal(
    rng: np.random.Generator, n: int, S_true: float, sigma: float = 1.0
) -> np.ndarray:
    mu = S_true * sigma
    return mu + sigma * rng.standard_normal(n)


def _simulate_garch11_t_from_params(
    rng: np.random.Generator,
    *,
    n: int,
    mu: float,
    omega: float,
    g_alpha: float,
    g_beta: float,
    nu: float,
    initial_variance: float,
    burn: int,
) -> np.ndarray:
    if omega <= 0:
        return np.full(n, np.nan)
    if g_alpha < 0 or g_beta < 0 or g_alpha + g_beta >= 1:
        return np.full(n, np.nan)
    if nu <= 2:
        return np.full(n, np.nan)

    total = int(burn) + int(n)
    h = np.empty(total, float)
    eps = np.empty(total, float)
    x = np.empty(total, float)

    z = rng.standard_t(df=nu, size=total) * np.sqrt((nu - 2.0) / nu)
    h[0] = max(float(initial_variance), 1e-12)
    eps[0] = np.sqrt(h[0]) * z[0]
    x[0] = mu + eps[0]

    for t in range(1, total):
        h[t] = max(omega + g_alpha * (eps[t - 1] ** 2) + g_beta * h[t - 1], 1e-12)
        eps[t] = np.sqrt(h[t]) * z[t]
        x[t] = mu + eps[t]

    return x[int(burn) :]


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
    return _simulate_garch11_t_from_params(
        rng,
        n=n,
        mu=mu,
        omega=omega,
        g_alpha=g_alpha,
        g_beta=g_beta,
        nu=nu,
        initial_variance=sigma_uncond**2,
        burn=burn,
    )


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


def _param_map(param_names: list[str], params: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(param_names, params, strict=True)}


def fit_model_to_data(dgp: str, x: np.ndarray, cfg: "Config") -> dict[str, Any]:
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1d array with at least 2 observations")

    mu_hat = float(np.nanmean(x))
    sigma_hat = float(np.nanstd(x, ddof=1))
    if not np.isfinite(sigma_hat) or sigma_hat <= 0:
        sigma_hat = np.nan

    if dgp == "iid_normal":
        return {"model": "iid_normal", "mu": mu_hat, "sigma": sigma_hat}

    if dgp == "garch11_t":
        _, res, params = sharpe_mc.fit_candidate(x, "garch11_t")
        params_arr = np.asarray(params, dtype=float)
        param_names = [str(name) for name in res.params.index]
        p = _param_map(param_names, params_arr)
        return {
            "model": "garch11_t",
            "mu": float(p["mu"]),
            "omega": float(p["omega"]),
            "g_alpha": float(p["alpha[1]"]),
            "g_beta": float(p["beta[1]"]),
            "nu": float(p["nu"]),
            "initial_variance": float(np.var(x, ddof=1)),
            "burn": int(cfg.burn),
        }

    raise ValueError(f"Unknown model for fit: {dgp}")


def simulate_from_fitted_model(
    rng: np.random.Generator, theta_hat: dict[str, Any], n: int
) -> np.ndarray:
    model = str(theta_hat["model"])

    if model == "iid_normal":
        mu = float(theta_hat["mu"])
        sigma = float(theta_hat["sigma"])
        if not np.isfinite(sigma) or sigma <= 0:
            return np.full(n, np.nan)
        return mu + sigma * rng.standard_normal(n)

    if model == "garch11_t":
        return _simulate_garch11_t_from_params(
            rng,
            n=n,
            mu=float(theta_hat["mu"]),
            omega=float(theta_hat["omega"]),
            g_alpha=float(theta_hat["g_alpha"]),
            g_beta=float(theta_hat["g_beta"]),
            nu=float(theta_hat["nu"]),
            initial_variance=float(theta_hat["initial_variance"]),
            burn=int(theta_hat["burn"]),
        )

    raise ValueError(f"Unknown fitted model: {model}")


def se_rep_parametric_bootstrap(
    dgp: str, x: np.ndarray, cfg: "Config", rng: np.random.Generator
) -> float:
    try:
        theta_hat = fit_model_to_data(dgp, x, cfg)
    except Exception:
        return np.nan

    B = max(2, int(cfg.B_rep))
    s_star = np.empty(B, float)
    for idx in range(B):
        try:
            x_star = simulate_from_fitted_model(rng, theta_hat=theta_hat, n=int(np.asarray(x).shape[0]))
            s_star[idx] = sharpe_hat(x_star)
        except Exception:
            s_star[idx] = np.nan

    s_star = s_star[np.isfinite(s_star)]
    if s_star.size < 2:
        return np.nan
    return float(np.std(s_star, ddof=1))


def se_iid_analytic(S_hat: float, n: int) -> float:
    return float(np.sqrt((1.0 + 0.5 * S_hat**2) / n))


def _coverage_value(lo: float, hi: float, S_true: float) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return float(lo <= S_true <= hi)


def _reject_zero_value(lo: float, hi: float) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return float((lo > 0.0) or (hi < 0.0))


def _aggregate_method_row(
    *,
    dgp: str,
    n: int,
    S_true: float,
    method: str,
    se_arr: np.ndarray,
    cov_arr: np.ndarray,
    rej_arr: np.ndarray,
    mc_sd: float,
) -> dict[str, Any]:
    mean_se = float(np.nanmean(se_arr)) if np.any(np.isfinite(se_arr)) else np.nan
    return {
        "dgp": dgp,
        "n": int(n),
        "S_true": float(S_true),
        "method": method,
        "coverage_95": float(np.nanmean(cov_arr)),
        "reject_rate_H0_S_eq_0": float(np.nanmean(rej_arr)),
        "se_ratio_meanSE_over_mcSD": (mean_se / mc_sd) if np.isfinite(mean_se) and mc_sd > 0 else np.nan,
        "fit_fail_rate": float(np.mean(np.isnan(se_arr))),
        "se_cell": np.nan,
        "omega_hat_cell": np.nan,
    }


@dataclass(frozen=True)
class Config:
    seed: int = 0
    alpha: float = 0.05
    R: int = 60025
    B_rep: int = 199
    B_cell: int | None = None
    dgps: tuple[str, ...] = ("iid_normal", "garch11_t")
    methods: tuple[str, ...] = (ANALYTIC_METHOD, BOOTSTRAP_METHOD)
    n_grid: tuple[int, ...] = (60, 120)
    S_grid: tuple[float, ...] = (0.0, 0.5, 1.0)
    g_alpha: float = 0.05
    g_beta: float = 0.90
    nu: float = 7.0
    burn: int = 500
    bootstrap_every_k: int = 1
    max_workers: int = max(1, (os.cpu_count() or 2) - 1)

    def __post_init__(self) -> None:
        if int(self.B_rep) < 99:
            raise ValueError("B_rep must be at least 99.")
        if int(self.bootstrap_every_k) < 1:
            raise ValueError("bootstrap_every_k must be at least 1.")
        unsupported_dgps = sorted(set(self.dgps) - set(SUPPORTED_DGPS))
        if unsupported_dgps:
            raise ValueError(f"Unsupported dgps: {unsupported_dgps}")
        unsupported_methods = sorted(set(self.methods) - set(SUPPORTED_METHODS))
        if unsupported_methods:
            raise ValueError(f"Unsupported methods: {unsupported_methods}")


def stable_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def run_cell(dgp: str, n: int, S_true: float, cfg_dict: dict) -> tuple[list[dict], dict]:
    cfg = Config(**cfg_dict)
    methods = set(cfg.methods)
    reps = max(1, int(cfg.R))
    bootstrap_every_k = max(1, int(cfg.bootstrap_every_k))
    rng_rep = np.random.default_rng(stable_seed(cfg.seed, "rep", dgp, n, S_true))

    s_hats = np.empty(reps, float)
    se_a = np.full(reps, np.nan, float)
    cov_a = np.full(reps, np.nan, float)
    rej_a = np.full(reps, np.nan, float)
    se_b = np.full(reps, np.nan, float)
    cov_b = np.full(reps, np.nan, float)
    rej_b = np.full(reps, np.nan, float)

    last_boot_se = np.nan

    for rep in range(reps):
        x = simulate_from_true_dgp(rng_rep, dgp=dgp, n=n, S_true=S_true, cfg=cfg)
        s_hat = sharpe_hat(x)
        s_hats[rep] = s_hat

        if ANALYTIC_METHOD in methods:
            se = se_iid_analytic(s_hat, n)
            lo, hi = wald_ci(s_hat, se, cfg.alpha)
            se_a[rep] = se
            cov_a[rep] = _coverage_value(lo, hi, S_true)
            rej_a[rep] = _reject_zero_value(lo, hi)

        if BOOTSTRAP_METHOD in methods:
            if (rep % bootstrap_every_k == 0) or not np.isfinite(last_boot_se):
                last_boot_se = se_rep_parametric_bootstrap(dgp=dgp, x=x, cfg=cfg, rng=rng_rep)
            lo, hi = wald_ci(s_hat, last_boot_se, cfg.alpha)
            se_b[rep] = last_boot_se
            cov_b[rep] = _coverage_value(lo, hi, S_true)
            rej_b[rep] = _reject_zero_value(lo, hi)

    finite_s = s_hats[np.isfinite(s_hats)]
    mc_sd = float(np.std(finite_s, ddof=1)) if finite_s.size > 1 else np.nan
    diagnostics_row = {
        "dgp": dgp,
        "n": int(n),
        "S_true": float(S_true),
        "bias": float(np.nanmean(s_hats - S_true)),
        "rmse": float(np.sqrt(np.nanmean((s_hats - S_true) ** 2))),
        "mc_sd_S_hat": mc_sd,
        "outer_reps": int(cfg.R),
        "B_rep": int(cfg.B_rep),
        "bootstrap_every_k": int(cfg.bootstrap_every_k),
    }

    method_rows: list[dict[str, Any]] = []
    if ANALYTIC_METHOD in methods:
        method_rows.append(
            _aggregate_method_row(
                dgp=dgp,
                n=n,
                S_true=S_true,
                method=ANALYTIC_METHOD,
                se_arr=se_a,
                cov_arr=cov_a,
                rej_arr=rej_a,
                mc_sd=mc_sd,
            )
        )
    if BOOTSTRAP_METHOD in methods:
        method_rows.append(
            _aggregate_method_row(
                dgp=dgp,
                n=n,
                S_true=S_true,
                method=BOOTSTRAP_METHOD,
                se_arr=se_b,
                cov_arr=cov_b,
                rej_arr=rej_b,
                mc_sd=mc_sd,
            )
        )
    return method_rows, diagnostics_row


def run_partA(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(dgp, n, s) for dgp in cfg.dgps for n in cfg.n_grid for s in cfg.S_grid]
    cfg_dict = asdict(cfg)

    method_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    if cfg.max_workers <= 1:
        for dgp, n, s_true in tasks:
            rows_m, row_d = run_cell(dgp, n, s_true, cfg_dict)
            method_rows.extend(rows_m)
            diagnostics_rows.append(row_d)
    else:
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {
                ex.submit(run_cell, dgp, n, s_true, cfg_dict): idx
                for idx, (dgp, n, s_true) in enumerate(tasks)
            }
            chunks: list[tuple[list[dict[str, Any]], dict[str, Any]] | None] = [None] * len(tasks)
            for fut in as_completed(futures):
                chunks[futures[fut]] = fut.result()
        for chunk in chunks:
            if chunk is None:
                raise RuntimeError("Missing worker result.")
            rows_m, row_d = chunk
            method_rows.extend(rows_m)
            diagnostics_rows.append(row_d)

    df_methods = pd.DataFrame(method_rows).sort_values(
        ["dgp", "n", "S_true", "method"]
    ).reset_index(drop=True)
    df_diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["dgp", "n", "S_true"]
    ).reset_index(drop=True)
    return df_methods, df_diagnostics


def _env_scalar(name: str, default: Any, caster) -> Any:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else caster(raw.strip())


def _env_tuple(name: str, default: tuple[Any, ...], caster) -> tuple[Any, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(caster(part) for part in parts)


def config_from_env() -> Config:
    return Config(
        seed=_env_scalar("SEED", 0, int),
        alpha=_env_scalar("ALPHA", 0.05, float),
        R=60025,
        B_rep=_env_scalar("B_REP", 199, int),
        B_cell=None,
        dgps=_env_tuple("DGPS", ("iid_normal", "garch11_t"), str),
        methods=(ANALYTIC_METHOD, BOOTSTRAP_METHOD),
        n_grid=_env_tuple("N_GRID", (60, 120), int),
        S_grid=_env_tuple("S_GRID", (0.0, 0.5, 1.0), float),
        g_alpha=_env_scalar("G_ALPHA", 0.05, float),
        g_beta=_env_scalar("G_BETA", 0.90, float),
        nu=_env_scalar("NU", 7.0, float),
        burn=_env_scalar("BURN", 500, int),
        bootstrap_every_k=_env_scalar("BOOTSTRAP_EVERY_K", 1, int),
        max_workers=_env_scalar("MAX_WORKERS", max(1, (os.cpu_count() or 2) - 1), int),
    )


if __name__ == "__main__":
    cfg = config_from_env()
    results, diagnostics = run_partA(cfg)

    run_dir = Path(os.environ.get("EXPERIMENT_BSC_OUTPUT_DIR", "outputs/experiment_bsc")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results_partA_cell_parametric_bootstrap.csv"
    diagnostics_path = run_dir / "results_partA_diagnostics.csv"

    results.to_csv(results_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print(results)
    print(diagnostics)
    print()
    print(f"Wrote: {results_path}")
    print(f"Wrote: {diagnostics_path}")
