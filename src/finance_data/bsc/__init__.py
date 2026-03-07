"""Public BSC simulation/plotting API."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .api import (
    ANALYTIC_METHOD,
    GARCH_MLE_METHOD,
    GARCH_ORACLE_METHOD,
    Config,
    ExportReportEntry,
    MainBundle,
    OracleBundle,
    default_config,
    plot_all,
    plot_grid,
    plot_oracle_coverage,
    plot_oracle_se,
    run,
    run_cached,
    run_ci_sweep_cached,
    run_garch11_oracle_analytic,
    run_main_bundle,
    run_oracle_bundle,
    write_plotly_png,
)
from .garch_oracle import h2_from_innov, omega_garch_closed_form, simulate_garch11
from .runtime import run_partA, run_partA_with_ci_sweep

if TYPE_CHECKING:
    from .eta import RuntimeEstimate, estimate_main_bundle_runtime

__all__ = [
    "ANALYTIC_METHOD",
    "GARCH_MLE_METHOD",
    "GARCH_ORACLE_METHOD",
    "Config",
    "ExportReportEntry",
    "MainBundle",
    "OracleBundle",
    "default_config",
    "run",
    "run_cached",
    "run_ci_sweep_cached",
    "run_partA",
    "run_partA_with_ci_sweep",
    "run_garch11_oracle_analytic",
    "run_main_bundle",
    "run_oracle_bundle",
    "RuntimeEstimate",
    "estimate_main_bundle_runtime",
    "plot_grid",
    "plot_all",
    "write_plotly_png",
    "plot_oracle_coverage",
    "plot_oracle_se",
    "h2_from_innov",
    "omega_garch_closed_form",
    "simulate_garch11",
]

_ETA_EXPORTS = {"RuntimeEstimate", "estimate_main_bundle_runtime"}


def __getattr__(name: str) -> Any:
    if name in _ETA_EXPORTS:
        module = importlib.import_module(".eta", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
