from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from . import plotting as plotting_module
from . import runtime as bsc_runtime
from . import garch_oracle as garch_oracle_module
from . import sharpe_mc as sharpe_mc_module
from .runtime import (
    ANALYTIC_METHOD,
    GARCH_MLE_METHOD,
    GARCH_ORACLE_METHOD,
    Config,
    run_partA,
    run_partA_with_ci_sweep,
)
from .garch_oracle import h2_from_innov, omega_garch_closed_form, simulate_garch11

__all__ = [
    "ANALYTIC_METHOD",
    "GARCH_MLE_METHOD",
    "GARCH_ORACLE_METHOD",
    "Config",
    "default_config",
    "run",
    "run_cached",
    "run_ci_sweep_cached",
    "run_garch11_oracle_analytic",
    "plot_grid",
    "plot_all",
    "write_plotly_png",
    "plot_oracle_coverage",
    "plot_oracle_se",
    "ExportReportEntry",
    "MainBundle",
    "OracleBundle",
    "run_main_bundle",
    "run_oracle_bundle",
    "h2_from_innov",
    "omega_garch_closed_form",
    "simulate_garch11",
]

if TYPE_CHECKING:
    from plotly.graph_objs import Figure

_tqdm: Any | None = None
_missing_notebook_progress_warned = False


def _has_supported_nbformat() -> bool:
    try:
        import nbformat
    except Exception:
        return False

    version = getattr(nbformat, "__version__", "")
    parts = version.split(".")
    parsed: list[int] = []
    for part in parts[:2]:
        try:
            parsed.append(int(part))
        except ValueError:
            return False
    if len(parsed) < 2:
        return False
    return tuple(parsed) >= (4, 2)


def _install_plotly_ipython_fallback(fig: Any) -> None:
    if _has_supported_nbformat():
        return

    try:
        from IPython.display import HTML, display
    except Exception:
        return

    def _safe_ipython_display() -> None:
        display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))

    fig._ipython_display_ = _safe_ipython_display


def _display_plotly_figure(fig: Any) -> None:
    _install_plotly_ipython_fallback(fig)

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
    except Exception:
        pass


def _is_truthy_env(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_interactive_session() -> bool:
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except Exception:
        pass

    try:
        return bool(sys.stderr.isatty() or sys.stdout.isatty())
    except Exception:
        return False


def _is_notebook_session() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return bool(shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell")
    except Exception:
        return False


def _has_ipywidgets() -> bool:
    try:
        import ipywidgets  # noqa: F401
    except Exception:
        return False
    return True


def _warn_missing_notebook_progress_backend_once() -> None:
    global _missing_notebook_progress_warned
    if _missing_notebook_progress_warned:
        return
    warnings.warn(
        "run_main_bundle progress is disabled in this notebook because the widget backend is unavailable. "
        "Install/enable ipywidgets and restart the notebook kernel.",
        UserWarning,
        stacklevel=3,
    )
    _missing_notebook_progress_warned = True


def _should_show_progress(progress: bool | None) -> bool:
    if progress is not None:
        return bool(progress)
    if _is_truthy_env("CI"):
        return False
    return _is_interactive_session()


def _resolve_tqdm() -> Any | None:
    global _tqdm
    if _tqdm is not None:
        return _tqdm
    try:
        from tqdm.auto import tqdm as tqdm_auto
    except Exception:
        try:
            from tqdm import tqdm as tqdm_std
        except Exception:
            return None
        _tqdm = tqdm_std
        return _tqdm
    _tqdm = tqdm_auto
    return _tqdm


def default_config(**overrides: Any) -> Config:
    """
    Notebook-friendly defaults for quick iteration.
    """
    cfg = Config(
        seed=0,
        alpha=0.05,
        R=1000,
        R_garch=500,
        dgps=("iid_normal", "garch11_t"),
        methods=(ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD),
        n_grid=(30, 60),
        S_grid=(-0.5, 0.0, 0.5),
        g_alpha=0.05,
        g_beta=0.90,
        garch_dist="t",
        nu=7.0,
        burn=500,
        max_workers=max(1, (os.cpu_count() or 2) - 1),
    )
    if not overrides:
        return cfg
    payload = cfg.__dict__.copy()
    payload.update(overrides)
    return Config(**payload)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_run_dir() -> Path:
    return (_project_root() / "outputs" / "bsc_final").resolve()


def _resolve_run_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    if output_dir is None:
        return _default_run_dir()
    out = Path(output_dir)
    if not out.is_absolute():
        out = _project_root() / out
    return out.resolve()


def _normalize_scope(scope: str) -> str:
    out = str(scope).strip()
    if not out:
        raise ValueError("scope must be a non-empty string.")
    return "".join(c if (c.isalnum() or c in {"_", "-"}) else "_" for c in out)


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _cache_payload(
    cfg: Config,
    scope: str,
    *,
    ci_levels: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    runtime_path = Path(bsc_runtime.__file__).resolve()
    sharpe_path = Path(sharpe_mc_module.__file__).resolve()
    oracle_path = Path(garch_oracle_module.__file__).resolve()
    payload: dict[str, Any] = {
        "cfg": asdict(cfg),
        "scope": scope,
        "sources": {
            "runtime_file": str(runtime_path),
            "runtime_sha": _sha12(runtime_path),
            "sharpe_mc_file": str(sharpe_path),
            "sharpe_mc_sha": _sha12(sharpe_path),
            "garch_oracle_file": str(oracle_path),
            "garch_oracle_sha": _sha12(oracle_path),
        },
    }
    if ci_levels is not None:
        payload["ci_levels"] = [float(level) for level in ci_levels]
    return payload


def _cache_hash(
    cfg: Config,
    scope: str,
    *,
    ci_levels: tuple[float, ...] | None = None,
) -> str:
    payload = _cache_payload(cfg, scope, ci_levels=ci_levels)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _normalize_ci_levels(ci_levels: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return bsc_runtime.normalize_ci_levels(ci_levels)


def _parquet_csv_paths(path_base: Path) -> tuple[Path, Path]:
    return path_base.with_suffix(".parquet"), path_base.with_suffix(".csv")


def _has_cache_artifact(path_base: Path) -> bool:
    parquet_path, csv_path = _parquet_csv_paths(path_base)
    return parquet_path.exists() or csv_path.exists()


def _read_cached_df(path_base: Path) -> tuple[pd.DataFrame, Path]:
    parquet_path, csv_path = _parquet_csv_paths(path_base)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), parquet_path
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path
    raise FileNotFoundError(f"No cache file found at {parquet_path} or {csv_path}.")


def _write_cached_df(df: pd.DataFrame, path_base: Path) -> Path:
    parquet_path, csv_path = _parquet_csv_paths(path_base)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path
    except Exception:
        if parquet_path.exists():
            parquet_path.unlink(missing_ok=True)
        df.to_csv(csv_path, index=False)
        return csv_path


def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the Part A Monte Carlo and return (results, diagnostics).
    """
    return run_partA(cfg)


def run_cached(
    cfg: Config,
    *,
    scope: str = "main",
    output_dir: str | os.PathLike[str] | None = None,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run Part A with a config+code-hash cache and return (results, diagnostics, cache_meta).
    """
    cache_scope = _normalize_scope(scope)
    run_dir = _resolve_run_dir(output_dir)
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_hash = _cache_hash(cfg, cache_scope)
    results_base = cache_dir / f"{cache_scope}_results_{cache_hash}"
    diagnostics_base = cache_dir / f"{cache_scope}_diagnostics_{cache_hash}"
    config_path = run_dir / f"config_{cache_scope}_{cache_hash}.json"

    cache_hit = False
    if not force_rerun and _has_cache_artifact(results_base) and _has_cache_artifact(diagnostics_base):
        try:
            results, results_path = _read_cached_df(results_base)
            diagnostics, diagnostics_path = _read_cached_df(diagnostics_base)
            cache_hit = True
        except Exception:
            cache_hit = False

    if not cache_hit:
        results, diagnostics = run_partA(cfg)
        results_path = _write_cached_df(results, results_base)
        diagnostics_path = _write_cached_df(diagnostics, diagnostics_base)

    payload = _cache_payload(cfg, cache_scope)
    payload["cache_hash"] = cache_hash
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cache_meta = {
        "cache_hit": bool(cache_hit),
        "cache_hash": cache_hash,
        "scope": cache_scope,
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "results_path": str(results_path),
        "diagnostics_path": str(diagnostics_path),
        "config_path": str(config_path),
    }
    return results, diagnostics, cache_meta


def run_ci_sweep_cached(
    cfg: Config,
    ci_levels: tuple[float, ...] | list[float],
    *,
    scope: str = "main",
    output_dir: str | os.PathLike[str] | None = None,
    force_rerun: bool = False,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run Part A + CI sweep with config/code/ci-level cache and return
    (results, diagnostics, ci_sweep, cache_meta).
    """
    normalized_levels = _normalize_ci_levels(ci_levels)
    cache_scope = _normalize_scope(scope)
    run_dir = _resolve_run_dir(output_dir)
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_hash = _cache_hash(cfg, cache_scope, ci_levels=normalized_levels)
    results_base = cache_dir / f"{cache_scope}_results_{cache_hash}"
    diagnostics_base = cache_dir / f"{cache_scope}_diagnostics_{cache_hash}"
    ci_sweep_base = cache_dir / f"{cache_scope}_ci_sweep_{cache_hash}"
    config_path = run_dir / f"config_{cache_scope}_{cache_hash}.json"

    cache_hit = False
    if (
        not force_rerun
        and _has_cache_artifact(results_base)
        and _has_cache_artifact(diagnostics_base)
        and _has_cache_artifact(ci_sweep_base)
    ):
        try:
            results, results_path = _read_cached_df(results_base)
            diagnostics, diagnostics_path = _read_cached_df(diagnostics_base)
            ci_sweep, ci_sweep_path = _read_cached_df(ci_sweep_base)
            cache_hit = True
        except Exception:
            cache_hit = False

    if not cache_hit:
        if progress_callback is None:
            results, diagnostics, ci_sweep = run_partA_with_ci_sweep(cfg, normalized_levels)
        else:
            results, diagnostics, ci_sweep = run_partA_with_ci_sweep(
                cfg,
                normalized_levels,
                progress_callback=progress_callback,
            )
        results_path = _write_cached_df(results, results_base)
        diagnostics_path = _write_cached_df(diagnostics, diagnostics_base)
        ci_sweep_path = _write_cached_df(ci_sweep, ci_sweep_base)

    payload = _cache_payload(cfg, cache_scope, ci_levels=normalized_levels)
    payload["cache_hash"] = cache_hash
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cache_meta = {
        "cache_hit": bool(cache_hit),
        "cache_hash": cache_hash,
        "scope": cache_scope,
        "ci_levels": list(normalized_levels),
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "results_path": str(results_path),
        "diagnostics_path": str(diagnostics_path),
        "ci_sweep_path": str(ci_sweep_path),
        "config_path": str(config_path),
    }
    return results, diagnostics, ci_sweep, cache_meta


def run_garch11_oracle_analytic(
    *,
    T: int = 240,
    S_true: float = 0.5,
    alpha1: float = 0.05,
    beta: float = 0.90,
    dist: str = "t",
    nu: float | None = 8.0,
    burn: int = 1000,
    R: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    API helper for the fast oracle-analytic GARCH(1,1) coverage run.
    """
    dist_name = str(dist).strip().lower()
    cfg = Config(
        seed=int(seed),
        alpha=float(alpha),
        R=int(R),
        R_garch=1,
        dgps=("garch11_t",),
        methods=(GARCH_ORACLE_METHOD,),
        n_grid=(int(T),),
        S_grid=(float(S_true),),
        g_alpha=float(alpha1),
        g_beta=float(beta),
        garch_dist=("normal" if dist_name in {"normal", "gaussian"} else "t"),
        nu=(float(nu) if nu is not None else 8.0),
        burn=int(burn),
        max_workers=1,
    )
    return run_partA(cfg)


def plot_oracle_coverage(*args, **kwargs):
    return plotting_module.plot_oracle_coverage(*args, **kwargs)


def plot_oracle_se(*args, **kwargs):
    return plotting_module.plot_oracle_se(*args, **kwargs)


def write_plotly_png(
    fig: Any,
    out_html: str | os.PathLike[str],
    *,
    width: int = 3200,
    height: int = 2000,
    scale: float = 2.0,
) -> Path:
    return plotting_module.write_plotly_png(fig, out_html, width=width, height=height, scale=scale)


def _apply_shared_axis_labels(fig: Any, *, x_label: str, y_label: str) -> None:
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)

    ann = list(fig.layout.annotations) if fig.layout.annotations else []
    ann.append(
        dict(
            text=x_label,
            x=0.5,
            y=-0.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="center",
            yanchor="top",
        )
    )
    ann.append(
        dict(
            text=y_label,
            x=-0.10,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            textangle=-90,
            xanchor="right",
            yanchor="middle",
        )
    )

    margin = fig.layout.margin
    m_l = int(getattr(margin, "l", 0) or 0)
    m_r = int(getattr(margin, "r", 0) or 0)
    m_t = int(getattr(margin, "t", 0) or 0)
    m_b = int(getattr(margin, "b", 0) or 0)
    fig.update_layout(
        title_text=None,
        annotations=ann,
        margin=dict(
            l=max(m_l, 110),
            r=max(m_r, 20),
            t=max(m_t, 20),
            b=max(m_b, 100),
        ),
    )


def plot_grid(
    results: pd.DataFrame,
    cfg: Config,
    *,
    metric: str,
    ylabel: str,
    baseline: float | None = None,
    title_suffix: str = "",
    methods: tuple[str, ...] | None = None,
) -> Any:
    """
    Plot one metric as a (dgp x method) grid directly in the notebook.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")

    required = {"dgp", "method", "n", "S_true", metric}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns for plot_grid: {missing}")

    frame = results.copy()
    frame["n_str"] = frame["n"].astype(str)
    methods_to_plot = tuple(methods) if methods is not None else tuple(cfg.methods)
    frame = frame[frame["method"].isin(methods_to_plot)]

    if frame.empty:
        raise ValueError("No rows available for selected methods in plot_grid.")

    vals = frame[metric].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size > 0:
        y_min = float(np.min(finite_vals))
        y_max = float(np.max(finite_vals))
    else:
        y_min, y_max = (0.0, 1.0)
    if baseline is not None:
        y_min = min(y_min, float(baseline))
        y_max = max(y_max, float(baseline))
    if np.isclose(y_min, y_max):
        y_min -= 0.01
        y_max += 0.01

    use_error_bars = {"mc_lo", "mc_hi", metric}.issubset(frame.columns)
    if use_error_bars:
        frame["err_y"] = (frame["mc_hi"] - frame[metric]).clip(lower=0.0)
        frame["err_y_minus"] = (frame[metric] - frame["mc_lo"]).clip(lower=0.0)

    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("plot_grid requires plotly. Install it with `pip install plotly`.") from exc

    line_kwargs: dict[str, Any] = {}
    if use_error_bars:
        line_kwargs["error_y"] = "err_y"
        line_kwargs["error_y_minus"] = "err_y_minus"

    n_order = [str(v) for v in sorted(frame["n"].dropna().astype(int).unique())]
    fig = px.line(
        frame.sort_values(["dgp", "method", "n", "S_true"]),
        x="S_true",
        y=metric,
        color="n_str",
        facet_row="dgp",
        facet_col="method",
        category_orders={"dgp": list(cfg.dgps), "method": list(methods_to_plot), "n_str": n_order},
        markers=True,
        **line_kwargs,
    )

    if baseline is not None:
        fig.add_hline(
            y=float(baseline),
            row="all",
            col="all",
            line_dash="dot",
            line_color="black",
            line_width=1.0,
        )

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(
        legend_title_text="n",
        margin=dict(l=50, r=20, t=20, b=50),
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("dgp=", "").replace("method=", "")))
    _apply_shared_axis_labels(fig, x_label="True Sharpe", y_label=ylabel)
    _install_plotly_ipython_fallback(fig)
    return fig


def _plot_coverage_vs_n_grid(
    results: pd.DataFrame,
    cfg: Config,
    *,
    metric: str = "coverage_95",
    ylabel: str = "95% coverage",
    baseline: float = 0.95,
) -> Any:
    """
    Plot coverage as y against sample size n as x with one line per true Sharpe.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")

    required = {"dgp", "method", "n", "S_true", metric}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns for coverage-vs-n plot: {missing}")

    frame = results.copy()
    frame = frame[frame["method"].isin(tuple(cfg.methods))]
    if frame.empty:
        raise ValueError("No rows available for selected methods in coverage-vs-n plot.")

    frame["n"] = pd.to_numeric(frame["n"], errors="coerce")
    if frame["n"].isna().any():
        raise ValueError("Column `n` must be numeric for coverage-vs-n plot.")
    frame["S_true_num"] = pd.to_numeric(frame["S_true"], errors="coerce")
    if frame["S_true_num"].isna().any():
        raise ValueError("Column `S_true` must be numeric for coverage-vs-n plot.")
    frame["S_true_str"] = frame["S_true_num"].map(lambda v: f"{float(v):g}")

    vals = frame[metric].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size > 0:
        y_min = float(np.min(finite_vals))
        y_max = float(np.max(finite_vals))
    else:
        y_min, y_max = (0.0, 1.0)
    y_min = min(y_min, float(baseline))
    y_max = max(y_max, float(baseline))
    if np.isclose(y_min, y_max):
        y_min -= 0.01
        y_max += 0.01

    use_error_bars = {"mc_lo", "mc_hi", metric}.issubset(frame.columns)
    if use_error_bars:
        frame["err_y"] = (frame["mc_hi"] - frame[metric]).clip(lower=0.0)
        frame["err_y_minus"] = (frame[metric] - frame["mc_lo"]).clip(lower=0.0)

    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("coverage-vs-n plot requires plotly. Install it with `pip install plotly`.") from exc

    line_kwargs: dict[str, Any] = {}
    if use_error_bars:
        line_kwargs["error_y"] = "err_y"
        line_kwargs["error_y_minus"] = "err_y_minus"

    s_order = [f"{float(v):g}" for v in sorted(frame["S_true_num"].dropna().unique())]
    fig = px.line(
        frame.sort_values(["dgp", "method", "S_true_num", "n"]),
        x="n",
        y=metric,
        color="S_true_str",
        facet_row="dgp",
        facet_col="method",
        category_orders={"dgp": list(cfg.dgps), "method": list(cfg.methods), "S_true_str": s_order},
        markers=True,
        **line_kwargs,
    )

    fig.add_hline(
        y=float(baseline),
        row="all",
        col="all",
        line_dash="dot",
        line_color="black",
        line_width=1.0,
    )

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(
        legend_title_text="True Sharpe",
        margin=dict(l=50, r=20, t=20, b=50),
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("dgp=", "").replace("method=", "")))
    _apply_shared_axis_labels(fig, x_label="n", y_label=ylabel)
    _install_plotly_ipython_fallback(fig)
    return fig


def plot_all(
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cfg: Config,
    *,
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Plot key Part A outputs inline in the notebook and return figure handles.
    """
    figs: dict[str, Any] = {}

    figs["coverage_95"] = plot_grid(
        results,
        cfg,
        metric="coverage_95",
        ylabel="95% coverage",
        baseline=0.95,
    )
    _display_plotly_figure(figs["coverage_95"])
    if progress_callback is not None:
        progress_callback()

    figs["coverage_95_vs_n"] = _plot_coverage_vs_n_grid(
        results,
        cfg,
        metric="coverage_95",
        ylabel="95% coverage",
        baseline=0.95,
    )
    _display_plotly_figure(figs["coverage_95_vs_n"])
    if progress_callback is not None:
        progress_callback()

    figs["reject_rate_H0_S_eq_0"] = plot_grid(
        results,
        cfg,
        metric="reject_rate_H0_S_eq_0",
        ylabel="Reject rate (H0: S = 0)",
        baseline=float(cfg.alpha),
    )
    _display_plotly_figure(figs["reject_rate_H0_S_eq_0"])
    if progress_callback is not None:
        progress_callback()

    for metric, label, baseline in (
        ("rmse", "RMSE", None),
        ("bias", "Bias", 0.0),
    ):
        frame = diagnostics.copy()
        frame["method"] = "__diagnostic__"
        figs[metric] = plot_grid(
            frame,
            cfg,
            metric=metric,
            ylabel=label,
            baseline=baseline,
            title_suffix=" | diagnostics",
            methods=("__diagnostic__",),
        )
        _display_plotly_figure(figs[metric])
        if progress_callback is not None:
            progress_callback()

    return figs


@dataclass(frozen=True)
class ExportReportEntry:
    name: str
    status: str
    png_path: str | None
    html_path: str | None
    warning: str | None


@dataclass(frozen=True)
class MainBundle:
    results: pd.DataFrame
    diagnostics: pd.DataFrame
    ci_sweep: pd.DataFrame
    cache_meta: dict[str, Any]
    figures: dict[str, Any]


@dataclass(frozen=True)
class OracleBundle:
    results: pd.DataFrame
    diagnostics: pd.DataFrame
    figures: dict[str, Any]
    export_report: dict[str, ExportReportEntry]
    cache_meta: dict[str, Any]


def _resolve_png_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like)
    if path.suffix.lower() == ".png":
        return path
    if path.suffix:
        return path.with_suffix(".png")
    return path.with_suffix(".png")


def _resolve_export_dir(
    export_dir: str | os.PathLike[str] | None,
    *,
    output_dir: str | os.PathLike[str] | None,
) -> Path:
    if export_dir is None:
        return _resolve_run_dir(output_dir)
    resolved = Path(export_dir)
    if not resolved.is_absolute():
        resolved = _project_root() / resolved
    return resolved.resolve()


def _export_plotly_with_fallback(
    *,
    name: str,
    fig: Any,
    png_target: str | os.PathLike[str],
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> ExportReportEntry:
    png_path = _resolve_png_path(png_target)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path = write_plotly_png(
            fig,
            png_path,
            width=png_width,
            height=png_height,
            scale=png_scale,
        )
    except Exception as exc:
        raise RuntimeError(f"PNG export failed for `{name}` at `{png_path}`: {exc}") from exc

    return ExportReportEntry(
        name=name,
        status="png",
        png_path=str(out_path),
        html_path=None,
        warning=None,
    )


def run_main_bundle(
    cfg: Config,
    *,
    ci_levels: tuple[float, ...] | list[float] = (0.90, 0.95, 0.975, 0.99),
    scope: str = "main",
    output_dir: str | os.PathLike[str] | None = None,
    force_rerun: bool = False,
    include_plots: bool = True,
    progress: bool | None = None,
    export_png: bool = True,
    export_dir: str | os.PathLike[str] | None = None,
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> MainBundle:
    notebook_session = _is_notebook_session()
    missing_notebook_backend = notebook_session and (not _has_ipywidgets())

    if missing_notebook_backend:
        message = (
            "run_main_bundle progress in notebooks requires ipywidgets. "
            "Install/enable ipywidgets and restart the notebook kernel."
        )
        if progress is True:
            raise ModuleNotFoundError(message)
        if progress is None:
            _warn_missing_notebook_progress_backend_once()
        show_progress = False
        tqdm_impl = None
    else:
        tqdm_impl = _resolve_tqdm()
        if progress is True and tqdm_impl is None:
            raise ModuleNotFoundError(
                "run_main_bundle(progress=True) requires tqdm. Install it with `pip install tqdm` "
                "and restart the notebook kernel."
            )
        show_progress = _should_show_progress(progress) and tqdm_impl is not None

    leave_bars = notebook_session
    stage_total = 1 + (1 if include_plots else 0)
    compute_total = len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    plot_total = 5

    stage_bar: Any | None = None
    compute_bar: Any | None = None
    plot_bar: Any | None = None
    compute_ticks = 0
    plot_ticks = 0

    def _tick_compute() -> None:
        nonlocal compute_ticks
        compute_ticks += 1
        if compute_bar is not None:
            compute_bar.update(1)

    def _tick_plot() -> None:
        nonlocal plot_ticks
        plot_ticks += 1
        if plot_bar is not None:
            plot_bar.update(1)

    if show_progress:
        stage_bar = tqdm_impl(total=stage_total, desc="main bundle", dynamic_ncols=True, leave=leave_bars, position=0)
        compute_bar = tqdm_impl(total=compute_total, desc="compute cells", dynamic_ncols=True, leave=leave_bars, position=1)
        if include_plots:
            plot_bar = tqdm_impl(total=plot_total, desc="plot steps", dynamic_ncols=True, leave=leave_bars, position=2)

    try:
        results, diagnostics, ci_sweep, cache_meta = run_ci_sweep_cached(
            cfg,
            ci_levels=ci_levels,
            scope=scope,
            output_dir=output_dir,
            force_rerun=force_rerun,
            progress_callback=_tick_compute if show_progress else None,
        )
        if compute_bar is not None and bool(cache_meta.get("cache_hit")):
            remaining = max(int(compute_total) - int(compute_ticks), 0)
            if remaining:
                compute_bar.update(remaining)
                compute_ticks += remaining
        if stage_bar is not None:
            stage_bar.update(1)

        if include_plots:
            figures: dict[str, Any] = plot_all(
                results,
                diagnostics,
                cfg,
                progress_callback=_tick_plot if show_progress else None,
            )
            if export_png:
                resolved_export_dir = _resolve_export_dir(export_dir, output_dir=output_dir)
                for fig_name, fig in figures.items():
                    _export_plotly_with_fallback(
                        name=fig_name,
                        fig=fig,
                        png_target=resolved_export_dir / f"{fig_name}.png",
                        png_width=png_width,
                        png_height=png_height,
                        png_scale=png_scale,
                    )
            if plot_bar is not None:
                remaining = max(int(plot_total) - int(plot_ticks), 0)
                if remaining:
                    plot_bar.update(remaining)
                    plot_ticks += remaining
            if stage_bar is not None:
                stage_bar.update(1)
        else:
            figures = {}
    finally:
        for bar in (plot_bar, compute_bar, stage_bar):
            if bar is not None:
                bar.close()

    return MainBundle(
        results=results,
        diagnostics=diagnostics,
        ci_sweep=ci_sweep,
        cache_meta=cache_meta,
        figures=figures,
    )


def run_oracle_bundle(
    cfg: Config,
    *,
    scope: str = "oracle_thesis",
    output_dir: str | os.PathLike[str] | None = None,
    force_rerun: bool = False,
    main_n_grid: tuple[int, ...] = (36, 60, 120, 360, 600),
    appendix_n_grid: tuple[int, ...] | None = None,
    main_y_range: tuple[float, float] = (0.93, 0.995),
    appendix_y_range: tuple[float, float] = (0.93, 1.0),
    export_dir: str | os.PathLike[str] | None = None,
    export_png: bool = True,
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> OracleBundle:
    if appendix_n_grid is None:
        appendix_n_grid = (12,) + tuple(main_n_grid)

    results, diagnostics, cache_meta = run_cached(
        cfg,
        scope=scope,
        output_dir=output_dir,
        force_rerun=force_rerun,
    )

    oracle_plot_df = results.rename(columns={"coverage_95": "coverage"})[
        ["n", "S_true", "coverage", "mc_lo", "mc_hi", "dgp"]
    ]
    oracle_main_plot_df = oracle_plot_df[oracle_plot_df["n"].isin(main_n_grid)].copy()
    oracle_appendix_plot_df = oracle_plot_df[oracle_plot_df["n"].isin(appendix_n_grid)].copy()
    oracle_se_plot_df = results[["n", "S_true", "se_cell", "dgp"]]
    oracle_se_plot_df = oracle_se_plot_df[oracle_se_plot_df["n"].isin(main_n_grid)].copy()

    fig_oracle_main = plot_oracle_coverage(oracle_main_plot_df, y_range=main_y_range)

    fig_oracle_se = plot_oracle_se(oracle_se_plot_df)

    fig_oracle_appendix = plot_oracle_coverage(oracle_appendix_plot_df, y_range=appendix_y_range)

    figures = {
        "oracle_coverage_main": fig_oracle_main,
        "oracle_se_main": fig_oracle_se,
        "oracle_coverage_appendix": fig_oracle_appendix,
    }

    export_report: dict[str, ExportReportEntry] = {}
    if export_png:
        resolved_export_dir = _resolve_export_dir(export_dir, output_dir=output_dir)
        export_report = {
            "oracle_coverage_main": _export_plotly_with_fallback(
                name="oracle_coverage_main",
                fig=fig_oracle_main,
                png_target=resolved_export_dir / "oracle_coverage_main.png",
                png_width=png_width,
                png_height=png_height,
                png_scale=png_scale,
            ),
            "oracle_se_main": _export_plotly_with_fallback(
                name="oracle_se_main",
                fig=fig_oracle_se,
                png_target=resolved_export_dir / "oracle_se_main.png",
                png_width=png_width,
                png_height=png_height,
                png_scale=png_scale,
            ),
            "oracle_coverage_appendix": _export_plotly_with_fallback(
                name="oracle_coverage_appendix",
                fig=fig_oracle_appendix,
                png_target=resolved_export_dir / "oracle_coverage_appendix.png",
                png_width=png_width,
                png_height=png_height,
                png_scale=png_scale,
            ),
        }

    return OracleBundle(
        results=results,
        diagnostics=diagnostics,
        figures=figures,
        export_report=export_report,
        cache_meta=cache_meta,
    )
