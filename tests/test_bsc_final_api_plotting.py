from pathlib import Path
import sys

import pandas as pd
from plotly.graph_objs import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import finance_data.bsc.api as api


def _sample_results() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for dgp in ("iid_normal", "garch11_t"):
        for method in (api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD):
            for n in (30, 60):
                for s_true in (-0.5, 0.0, 0.5):
                    coverage = 0.92 + (0.01 if n == 60 else 0.0) - 0.01 * abs(s_true)
                    rows.append(
                        {
                            "dgp": dgp,
                            "method": method,
                            "n": n,
                            "S_true": s_true,
                            "coverage_95": coverage,
                            "mc_lo": coverage - 0.02,
                            "mc_hi": coverage + 0.02,
                            "reject_rate_H0_S_eq_0": 0.05 + (0.02 if s_true != 0.0 else 0.0),
                        }
                    )
    return pd.DataFrame(rows)


def _sample_diagnostics() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for dgp in ("iid_normal", "garch11_t"):
        for n in (30, 60):
            for s_true in (-0.5, 0.0, 0.5):
                rows.append(
                    {
                        "dgp": dgp,
                        "n": n,
                        "S_true": s_true,
                        "bias": 0.01 * s_true,
                        "rmse": 0.10 + 0.01 * (n == 30),
                    }
                )
    return pd.DataFrame(rows)


def test_plot_all_returns_plotly_figures_for_all_metrics() -> None:
    cfg = api.default_config(
        dgps=("iid_normal", "garch11_t"),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD),
        n_grid=(30, 60),
        S_grid=(-0.5, 0.0, 0.5),
        max_workers=1,
    )

    figs = api.plot_all(_sample_results(), _sample_diagnostics(), cfg)

    assert set(figs.keys()) == {"coverage_95", "reject_rate_H0_S_eq_0", "rmse", "bias"}
    assert all(isinstance(fig, Figure) for fig in figs.values())


def test_plot_grid_coverage_uses_error_bars_when_mc_bounds_exist() -> None:
    cfg = api.default_config(
        dgps=("iid_normal", "garch11_t"),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD),
        n_grid=(30, 60),
        S_grid=(-0.5, 0.0, 0.5),
        max_workers=1,
    )

    fig = api.plot_grid(
        _sample_results(),
        cfg,
        metric="coverage_95",
        ylabel="95% coverage",
        baseline=0.95,
    )

    assert isinstance(fig, Figure)
    assert any(trace.error_y is not None and trace.error_y.array is not None for trace in fig.data)


def test_plot_grid_noncoverage_metric_renders_without_error_bars() -> None:
    cfg = api.default_config(
        dgps=("iid_normal", "garch11_t"),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD),
        n_grid=(30, 60),
        S_grid=(-0.5, 0.0, 0.5),
        max_workers=1,
    )

    diagnostics = _sample_diagnostics().copy()
    diagnostics["method"] = "__diagnostic__"

    fig = api.plot_grid(
        diagnostics,
        cfg,
        metric="rmse",
        ylabel="RMSE",
        methods=("__diagnostic__",),
    )

    assert isinstance(fig, Figure)
    assert all(trace.error_y is None or trace.error_y.array is None for trace in fig.data)


def test_plot_oracle_se_wrapper_resolves_function() -> None:
    df = pd.DataFrame(
        {
            "dgp": ["garch11_t"] * 6,
            "n": [36, 36, 36, 60, 60, 60],
            "S_true": [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
            "se_cell": [0.11, 0.10, 0.11, 0.09, 0.085, 0.09],
        }
    )
    fig = api.plot_oracle_se(df)
    assert isinstance(fig, Figure)
