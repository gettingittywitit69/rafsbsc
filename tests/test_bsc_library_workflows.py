from pathlib import Path
import importlib
import sys
import warnings

import pandas as pd
from plotly.graph_objs import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import finance_data.bsc.api as api


def _sample_oracle_results() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n in (12, 36, 60):
        for s_true in (-0.5, 0.0, 0.5):
            coverage = 0.94 + 0.005 * (n >= 36) - 0.005 * abs(s_true)
            rows.append(
                {
                    "dgp": "garch11_t",
                    "method": api.GARCH_ORACLE_METHOD,
                    "n": n,
                    "S_true": s_true,
                    "coverage_95": coverage,
                    "mc_lo": coverage - 0.01,
                    "mc_hi": coverage + 0.01,
                    "se_cell": 0.10 + 0.01 * (n == 12),
                }
            )
    return pd.DataFrame(rows)


def test_run_main_bundle_returns_dataclass_payload(monkeypatch) -> None:
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": True, "cache_hash": "abc123"}

    monkeypatch.setattr(api, "run_ci_sweep_cached", lambda *args, **kwargs: (results, diagnostics, ci_sweep, cache_meta))
    monkeypatch.setattr(api, "plot_all", lambda *args, **kwargs: {"coverage_95": Figure()})

    bundle = api.run_main_bundle(api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1))

    assert isinstance(bundle, api.MainBundle)
    assert bundle.results.equals(results)
    assert bundle.diagnostics.equals(diagnostics)
    assert bundle.ci_sweep.equals(ci_sweep)
    assert bundle.cache_meta == cache_meta
    assert "coverage_95" in bundle.figures


def test_run_oracle_bundle_falls_back_to_html_when_png_export_fails(tmp_path, monkeypatch) -> None:
    results = _sample_oracle_results()
    diagnostics = pd.DataFrame([{"dgp": "garch11_t", "n": 36, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    cache_meta = {"cache_hit": True, "cache_hash": "oracle123"}

    monkeypatch.setattr(api, "run_cached", lambda *args, **kwargs: (results, diagnostics, cache_meta))
    monkeypatch.setattr(api, "plot_oracle_coverage", lambda *args, **kwargs: Figure())
    monkeypatch.setattr(api, "plot_oracle_se", lambda *args, **kwargs: Figure())
    monkeypatch.setattr(api, "write_plotly_png", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("png failed")))

    bundle = api.run_oracle_bundle(
        api.default_config(
            R=10,
            R_garch=10,
            dgps=("garch11_t",),
            methods=(api.GARCH_ORACLE_METHOD,),
            n_grid=(12, 36, 60),
            S_grid=(-0.5, 0.0, 0.5),
            max_workers=1,
        ),
        main_n_grid=(36, 60),
        appendix_n_grid=(12, 36, 60),
        export_dir=tmp_path,
    )

    assert isinstance(bundle, api.OracleBundle)
    assert set(bundle.export_report.keys()) == {"oracle_coverage_main", "oracle_se_main", "oracle_coverage_appendix"}
    for key, entry in bundle.export_report.items():
        assert entry.status == "html_fallback", key
        assert entry.warning is not None and "png failed" in entry.warning
        assert entry.html_path is not None
        assert Path(entry.html_path).exists()


def test_legacy_shim_imports_warn_and_reexport() -> None:
    sys.path.insert(0, str(ROOT))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_api = importlib.reload(importlib.import_module("src.bsc_final_api"))
        legacy_plot = importlib.reload(importlib.import_module("src.plot_oracle_coverage"))

    assert hasattr(legacy_api, "run_cached")
    assert hasattr(legacy_plot, "write_plotly_png")
    assert any("deprecated" in str(w.message).lower() for w in caught)
