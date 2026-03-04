from __future__ import annotations

import csv
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, TypedDict

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.stats import kurtosis, norm, skew

from arch.univariate import ARX, ConstantMean, ConstantVariance, GARCH, Normal, StudentsT
from arch.utility.exceptions import StartingValueWarning

DGP_LIST: tuple[str, ...] = ("iid_normal", "iid_t5", "ar1_t5", "garch11_t5")
N_GRID: tuple[int, ...] = (120, 240, 1200)
S_TRUE_GRID: tuple[float, ...] = (0.0, 0.5,1)
DEFAULT_REPS = 20000
DEFAULT_SIGMA = 0.04
DEFAULT_DF = 5
DEFAULT_PHI = 0.3
DEFAULT_ALPHA = 0.05  # GARCH alpha
DEFAULT_BETA = 0.90
DEFAULT_BURN = 500
DEFAULT_N_TRIALS = 24

Z_95 = 1.959963984540054
Z_90_ONE_SIDED = 1.6448536269


class SummaryRow(TypedDict):
    dgp: str
    n: int
    S_true: float
    method: str
    bias: float
    rmse: float
    coverage_95: float
    reject_rate_H0_S_le_0: float
    se_ratio: float
    psr_reject_rate: float
    dsr_reject_rate: float


# ---------- helpers ----------

def _ensure_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Non-finite values in {name}")


def _auto_bandwidth(n: int) -> int:
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def _standardized_t(df: int, size: int, rng: Generator) -> np.ndarray:
    scale = math.sqrt(df / (df - 2))
    return rng.standard_t(df, size=size) / scale


def _spawn_seeds(master_seed: int | None, n_cells: int) -> list[int | None]:
    if master_seed is None:
        return [None] * n_cells
    root = SeedSequence(master_seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in root.spawn(n_cells)]


# ---------- DGPs ----------

def simulate_iid_normal(n: int, mu: float, sigma: float, rng: Generator) -> np.ndarray:
    return mu + sigma * rng.standard_normal(n)


def simulate_iid_t5(n: int, mu: float, sigma: float, rng: Generator, df: int = DEFAULT_DF) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must exceed 2")
    z = _standardized_t(df, n, rng)
    return mu + sigma * z


def simulate_ar1_t5(
    n: int,
    mu: float,
    sigma: float,
    rng: Generator,
    phi: float = DEFAULT_PHI,
    df: int = DEFAULT_DF,
    burn: int = DEFAULT_BURN,
) -> np.ndarray:
    if abs(phi) >= 1:
        raise ValueError("|phi| must be < 1")
    total = n + burn
    shocks = _standardized_t(df, total, rng)
    sigma_e = sigma * math.sqrt(1 - phi**2)
    x = np.empty(total)
    x[0] = mu
    for t in range(1, total):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma_e * shocks[t]
    series = x[burn:]
    _ensure_finite(series, "ar1_t5")
    return series


def simulate_garch11_t5(
    n: int,
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
    df: int,
    burn: int,
) -> np.ndarray:
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("alpha,beta >=0 and alpha+beta<1")
    if df <= 2:
        raise ValueError("df must exceed 2")
    omega = sigma**2 * (1 - alpha - beta)
    if omega <= 0:
        raise ValueError("omega must be positive")

    model = ConstantMean()
    model.volatility = GARCH(p=1, q=1)
    model.distribution = StudentsT()
    params = np.array([mu, omega, alpha, beta, df], dtype=float)
    sim = model.simulate(params, nobs=n, burn=burn, initial_value_vol=sigma**2)
    series = np.asarray(sim["data"], dtype=float)
    if series.shape[0] != n:
        raise RuntimeError("Incorrect series length from arch simulate")
    _ensure_finite(series, "garch11_t5")
    return series


def simulate_dgp(
    dgp: str,
    n: int,
    S_true: float,
    *,
    sigma: float = DEFAULT_SIGMA,
    df: int = DEFAULT_DF,
    phi: float = DEFAULT_PHI,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    burn: int = DEFAULT_BURN,
    rng: Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if n < 10:
        raise ValueError("n must be at least 10")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if rng is None:
        if seed is not None:
            np.random.seed(seed)
        rng = default_rng(seed)
    mu = S_true * sigma
    if dgp == "iid_normal":
        series = simulate_iid_normal(n, mu, sigma, rng)
    elif dgp == "iid_t5":
        series = simulate_iid_t5(n, mu, sigma, rng, df=df)
    elif dgp == "ar1_t5":
        series = simulate_ar1_t5(n, mu, sigma, rng, phi=phi, df=df, burn=burn)
    elif dgp == "garch11_t5":
        series = simulate_garch11_t5(n, mu, sigma, alpha=alpha, beta=beta, df=df, burn=burn)
    else:
        raise ValueError(f"Unknown dgp {dgp}")
    if series.shape[0] != n:
        raise RuntimeError("Incorrect series length")
    _ensure_finite(series, "simulated series")
    return series


def fit_candidate(
    train: np.ndarray,
    name: str,
    starting_values: np.ndarray | None = None,
    maxiter: int = 500,
):
    train_arr = np.asarray(train, dtype=float)
    if train_arr.ndim != 1:
        raise ValueError("train must be a 1d array")
    if train_arr.size < 10:
        raise ValueError("train must have at least 10 observations")
    _ensure_finite(train_arr, "train")

    if name == "iid_normal":
        model = ConstantMean(train_arr)
        model.volatility = ConstantVariance()
        model.distribution = Normal()
    elif name == "iid_t":
        model = ConstantMean(train_arr)
        model.volatility = ConstantVariance()
        model.distribution = StudentsT()
    elif name == "garch11_t":
        model = ConstantMean(train_arr)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = StudentsT()
    elif name == "garch11_normal":
        model = ConstantMean(train_arr)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = Normal()
    elif name == "ar1_garch11_t":
        model = ARX(train_arr, lags=1)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = StudentsT()
    else:
        raise ValueError(f"Unknown candidate model {name}")

    fit_kwargs = {
        "disp": "off",
        "tol": 1e-6,
        "options": {"maxiter": int(maxiter)},
    }
    try:
        if starting_values is None:
            res = model.fit(**fit_kwargs)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", StartingValueWarning)
                res = model.fit(starting_values=starting_values, **fit_kwargs)
            if any(issubclass(w.category, StartingValueWarning) for w in caught):
                raise ValueError("starting values were rejected")
    except Exception:
        if starting_values is None:
            raise
        res = model.fit(
            disp="off",
            tol=1e-6,
            options={"maxiter": 500},
        )
    return model, res, np.asarray(res.params, dtype=float)


def simulate_from_fit(
    model,
    params,
    n: int,
    burn: int = 500,
    initial_value_vol: float | None = None,
) -> np.ndarray:
    sim = model.simulate(params, nobs=n, burn=burn, initial_value_vol=initial_value_vol)
    series = np.asarray(sim["data"], dtype=float)
    if series.shape[0] != n:
        raise RuntimeError("Incorrect series length from fitted model simulate")
    _ensure_finite(series, "simulate_from_fit")
    return series


# ---------- estimation ----------

def sharpe_ratio(x: np.ndarray) -> tuple[float, float, float]:
    mean_x = float(np.mean(x))
    std_x = float(np.std(x, ddof=1))
    if std_x <= 0:
        raise RuntimeError("Sample sd non-positive")
    return mean_x / std_x, mean_x, std_x


def se_naive(S_hat: float, n: int) -> float:
    return math.sqrt((1 + 0.5 * S_hat**2) / n)


def se_hac(x: np.ndarray, S_hat: float, bandwidth: int | None = None) -> float:
    n = x.shape[0]
    bw = _auto_bandwidth(n) if bandwidth is None else int(bandwidth)
    bw = max(0, min(bw, n - 1))
    mean_x = float(np.mean(x))
    std_x = float(np.std(x, ddof=1))
    y = (x - mean_x) / std_x
    psi = y - 0.5 * S_hat * (y**2 - 1)
    psi_c = psi - float(np.mean(psi))
    omega = float(np.mean(psi_c * psi_c))
    for k in range(1, bw + 1):
        gamma_k = float(np.mean(psi_c[:-k] * psi_c[k:]))
        weight = 1.0 - k / (bw + 1)
        omega += 2 * weight * gamma_k
    return math.sqrt(max(omega, 1e-12) / n)


def psr_probability(S_hat: float, skew_r: float, kurt_r: float, n: int, S_ref: float = 0.0) -> float:
    if n <= 1:
        return 0.5
    denom = math.sqrt(max(1 - skew_r * S_hat + 0.25 * (kurt_r - 1) * S_hat**2, 1e-12))
    z = (S_hat - S_ref) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def dsr_probability(
    S_hat: float,
    skew_r: float,
    kurt_r: float,
    n: int,
    n_trials: int,
    S_ref: float = 0.0,
) -> float:
    if n <= 1:
        return 0.5
    if n_trials <= 1:
        raise ValueError("n_trials must exceed 1")
    var_sr = max((1 - skew_r * S_hat + 0.25 * (kurt_r - 1) * S_hat**2) / (n - 1), 1e-12)
    sr_std = math.sqrt(var_sr)
    sr_star = S_ref + sr_std * norm.ppf(1 - 1.0 / n_trials)
    z = (S_hat - sr_star) / sr_std
    return float(norm.cdf(z))


def _coverage_and_reject(S_true: float, S_hat: float, se: float) -> tuple[bool, bool]:
    if se <= 0 or not math.isfinite(se):
        return False, False
    ci_low, ci_high = S_hat - Z_95 * se, S_hat + Z_95 * se
    reject = (S_hat / se) > Z_90_ONE_SIDED
    return (ci_low <= S_true <= ci_high), reject


# ---------- Monte Carlo ----------

def _summarize(
    *,
    dgp: str,
    n: int,
    S_true: float,
    method: str,
    s_hat: np.ndarray,
    coverage: np.ndarray,
    reject: np.ndarray,
    se_arr: np.ndarray,
    psr_reject_rate: float,
    dsr_reject_rate: float,
) -> SummaryRow:
    std_s_hat = float(np.std(s_hat, ddof=1))
    se_ratio = float(np.mean(se_arr) / std_s_hat) if std_s_hat > 0 else float("nan")
    return {
        "dgp": dgp,
        "n": int(n),
        "S_true": float(S_true),
        "method": method,
        "bias": float(np.mean(s_hat - S_true)),
        "rmse": float(np.sqrt(np.mean((s_hat - S_true) ** 2))),
        "coverage_95": float(np.mean(coverage)),
        "reject_rate_H0_S_le_0": float(np.mean(reject)),
        "se_ratio": se_ratio,
        "psr_reject_rate": psr_reject_rate,
        "dsr_reject_rate": dsr_reject_rate,
    }


def run_cell(
    dgp: str,
    n: int,
    S_true: float,
    *,
    reps: int = DEFAULT_REPS,
    seed: int | None = None,
    sigma: float = DEFAULT_SIGMA,
    df: int = DEFAULT_DF,
    phi: float = DEFAULT_PHI,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    burn: int = DEFAULT_BURN,
    n_trials: int = DEFAULT_N_TRIALS,
) -> List[SummaryRow]:
    if reps < 1:
        raise ValueError("reps must be positive")
    if n_trials <= 1:
        raise ValueError("n_trials must exceed 1")

    rng = default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    s_hat = np.empty(reps)
    se_naive_arr = np.empty(reps)
    se_hac_arr = np.empty(reps)
    cov_naive = np.empty(reps, dtype=bool)
    cov_hac = np.empty(reps, dtype=bool)
    rej_naive = np.empty(reps, dtype=bool)
    rej_hac = np.empty(reps, dtype=bool)
    psr_vals = np.empty(reps)
    dsr_vals = np.empty(reps)

    for i in range(reps):
        series = simulate_dgp(
            dgp,
            n,
            S_true,
            sigma=sigma,
            df=df,
            phi=phi,
            alpha=alpha,
            beta=beta,
            burn=burn,
            rng=rng,
        )
        S_est, _, _ = sharpe_ratio(series)
        s_hat[i] = S_est
        se_n = se_naive(S_est, n)
        se_r = se_hac(series, S_est)
        se_naive_arr[i] = se_n
        se_hac_arr[i] = se_r
        cov_naive[i], rej_naive[i] = _coverage_and_reject(S_true, S_est, se_n)
        cov_hac[i], rej_hac[i] = _coverage_and_reject(S_true, S_est, se_r)
        skew_x = float(skew(series, bias=False))
        kurt_x = float(kurtosis(series, fisher=False, bias=False))
        psr_vals[i] = psr_probability(S_est, skew_x, kurt_x, n)
        dsr_vals[i] = dsr_probability(S_est, skew_x, kurt_x, n, n_trials)

    psr_reject_rate = float(np.mean(psr_vals > 0.95))
    dsr_reject_rate = float(np.mean(dsr_vals > 0.95))

    rows: list[SummaryRow] = []
    rows.append(
        _summarize(
            dgp=dgp,
            n=n,
            S_true=S_true,
            method="naive_asymptotic",
            s_hat=s_hat,
            coverage=cov_naive,
            reject=rej_naive,
            se_arr=se_naive_arr,
            psr_reject_rate=psr_reject_rate,
            dsr_reject_rate=dsr_reject_rate,
        )
    )
    rows.append(
        _summarize(
            dgp=dgp,
            n=n,
            S_true=S_true,
            method="robust_hac",
            s_hat=s_hat,
            coverage=cov_hac,
            reject=rej_hac,
            se_arr=se_hac_arr,
            psr_reject_rate=psr_reject_rate,
            dsr_reject_rate=dsr_reject_rate,
        )
    )
    rows.append(
        _summarize(
            dgp=dgp,
            n=n,
            S_true=S_true,
            method="psr",
            s_hat=s_hat,
            coverage=cov_naive,
            reject=psr_vals > 0.95,
            se_arr=se_naive_arr,
            psr_reject_rate=psr_reject_rate,
            dsr_reject_rate=dsr_reject_rate,
        )
    )
    rows.append(
        _summarize(
            dgp=dgp,
            n=n,
            S_true=S_true,
            method="dsr",
            s_hat=s_hat,
            coverage=cov_naive,
            reject=dsr_vals > 0.95,
            se_arr=se_naive_arr,
            psr_reject_rate=psr_reject_rate,
            dsr_reject_rate=dsr_reject_rate,
        )
    )
    return rows


def run_experiment(
    *,
    dgps: Sequence[str] = DGP_LIST,
    n_grid: Sequence[int] = N_GRID,
    s_true_grid: Sequence[float] = S_TRUE_GRID,
    reps: int = DEFAULT_REPS,
    seed: int | None = None,
    sigma: float = DEFAULT_SIGMA,
    df: int = DEFAULT_DF,
    phi: float = DEFAULT_PHI,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    burn: int = DEFAULT_BURN,
    n_trials: int = DEFAULT_N_TRIALS,
    max_workers: int | None = None,
    run_sanity: bool = True,
) -> List[SummaryRow]:
    if run_sanity:
        sanity_seed = None if seed is None else seed + 10_000
        sanity_check_dgps(dgps, sigma=sigma, df=df, phi=phi, alpha=alpha, beta=beta, burn=burn, seed=sanity_seed)

    cells = list(product(dgps, n_grid, s_true_grid))
    seeds = _spawn_seeds(seed, len(cells))

    if max_workers == 1:
        chunks = [
            run_cell(
                dgp,
                n,
                s_true,
                reps=reps,
                seed=seed_i,
                sigma=sigma,
                df=df,
                phi=phi,
                alpha=alpha,
                beta=beta,
                burn=burn,
                n_trials=n_trials,
            )
            for (dgp, n, s_true), seed_i in zip(cells, seeds)
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_cell,
                    dgp,
                    n,
                    s_true,
                    reps=reps,
                    seed=seed_i,
                    sigma=sigma,
                    df=df,
                    phi=phi,
                    alpha=alpha,
                    beta=beta,
                    burn=burn,
                    n_trials=n_trials,
                ): idx
                for idx, ((dgp, n, s_true), seed_i) in enumerate(zip(cells, seeds))
            }
            chunks: list[list[SummaryRow] | None] = [None] * len(cells)
            for fut in as_completed(futures):
                idx = futures[fut]
                chunks[idx] = fut.result()
        if any(chunk is None for chunk in chunks):
            raise RuntimeError("Missing results from worker")
    results: list[SummaryRow] = []
    for chunk in chunks:
        if chunk is not None:
            results.extend(chunk)
    return results


def write_summary_csv(rows: Iterable[SummaryRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dgp",
        "n",
        "S_true",
        "method",
        "bias",
        "rmse",
        "coverage_95",
        "reject_rate_H0_S_le_0",
        "se_ratio",
        "psr_reject_rate",
        "dsr_reject_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------- sanity ----------

def sanity_check_dgps(
    dgps: Sequence[str] = DGP_LIST,
    *,
    sigma: float = DEFAULT_SIGMA,
    df: int = DEFAULT_DF,
    phi: float = DEFAULT_PHI,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    burn: int = DEFAULT_BURN,
    seed: int | None = None,
    n: int = 10_000,
    tol_mean: float = 0.10,
    tol_var: float = 0.20,
) -> None:
    rng = default_rng(seed)
    mu_target = 0.5 * sigma
    var_target = sigma**2
    mean_tol = tol_mean * max(abs(mu_target), sigma)
    var_tol = tol_var * var_target

    for dgp in dgps:
        series = simulate_dgp(
            dgp,
            n,
            0.5,
            sigma=sigma,
            df=df,
            phi=phi,
            alpha=alpha,
            beta=beta,
            burn=burn,
            rng=rng,
        )
        _ensure_finite(series, f"sanity {dgp}")
        mean_x = float(np.mean(series))
        var_x = float(np.var(series, ddof=1))
        if abs(mean_x - mu_target) > mean_tol:
            raise RuntimeError(f"Mean off for {dgp}: {mean_x:.4f} vs {mu_target:.4f}")
        if abs(var_x - var_target) > var_tol:
            raise RuntimeError(f"Variance off for {dgp}: {var_x:.6f} vs {var_target:.6f}")


# ---------- utilities ----------

def required_reps(alpha: float = 0.05, epsilon: float = 0.01, p: float = 0.5, m_cells: int | None = None) -> int:
    """Normal-approx sample size for a binomial proportion."""
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    alpha_cell = alpha if m_cells is None else alpha / m_cells
    z = norm.ppf(1 - alpha_cell / 2)
    R = z**2 * p * (1 - p) / (epsilon**2)
    return int(math.ceil(R))


def achieved_half_width(R: int, alpha: float = 0.05, p: float = 0.5, m_cells: int | None = None) -> float:
    if R <= 0:
        raise ValueError("R must be positive")
    alpha_cell = alpha if m_cells is None else alpha / m_cells
    z = norm.ppf(1 - alpha_cell / 2)
    return float(z * math.sqrt(p * (1 - p) / R))


__all__ = [
    "DGP_LIST",
    "N_GRID",
    "S_TRUE_GRID",
    "DEFAULT_REPS",
    "DEFAULT_SIGMA",
    "DEFAULT_DF",
    "DEFAULT_PHI",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_BURN",
    "DEFAULT_N_TRIALS",
    "run_cell",
    "run_experiment",
    "write_summary_csv",
    "simulate_dgp",
    "simulate_iid_normal",
    "simulate_iid_t5",
    "simulate_ar1_t5",
    "simulate_garch11_t5",
    "fit_candidate",
    "simulate_from_fit",
    "sharpe_ratio",
    "se_naive",
    "se_hac",
    "psr_probability",
    "dsr_probability",
    "sanity_check_dgps",
    "required_reps",
    "achieved_half_width",
    "SummaryRow",
]
