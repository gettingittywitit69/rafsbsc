from pathlib import Path
import sys

import pandas as pd
from plotly.basedatatypes import BaseFigure
from plotly.graph_objs import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_data.bsc.plotting import plot_oracle_coverage, plot_oracle_se, write_plotly_png


def _annotation_texts(fig: Figure) -> set[str]:
    return {str(a.text) for a in (fig.layout.annotations or ())}


def _stub_write_image(monkeypatch) -> list[tuple[Path, dict[str, object]]]:
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_write_image(self, path, *args, **kwargs) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        calls.append((out_path, dict(kwargs)))

    monkeypatch.setattr(BaseFigure, "write_image", fake_write_image)
    return calls


def test_plot_oracle_coverage_smoke(tmp_path, monkeypatch) -> None:
    calls = _stub_write_image(monkeypatch)
    df = pd.DataFrame(
        {
            "n": [120, 120, 120, 240, 240, 240],
            "S_true": [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
            "coverage": [0.93, 0.95, 0.94, 0.95, 0.96, 0.95],
            "mc_lo": [0.90, 0.92, 0.91, 0.92, 0.93, 0.92],
            "mc_hi": [0.96, 0.98, 0.97, 0.98, 0.99, 0.98],
        }
    )
    out_html = tmp_path / "test.html"
    fig = plot_oracle_coverage(df, out_html=str(out_html))
    assert fig is not None
    assert [p for p, _ in calls] == [out_html.with_suffix(".png")]
    assert out_html.with_suffix(".png").exists()
    assert not out_html.exists()
    assert fig.layout.legend.title.text == "n"
    assert fig.layout.xaxis.title.text in (None, "")
    assert fig.layout.yaxis.title.text in (None, "")
    ann = _annotation_texts(fig)
    assert "True Sharpe" in ann
    assert "95% Coverage" in ann


def test_plot_oracle_coverage_accepts_legacy_t_alias() -> None:
    df = pd.DataFrame(
        {
            "T": [120, 120, 240, 240],
            "S_true": [-0.5, 0.5, -0.5, 0.5],
            "coverage": [0.93, 0.94, 0.95, 0.96],
            "mc_lo": [0.90, 0.91, 0.92, 0.93],
            "mc_hi": [0.96, 0.97, 0.98, 0.99],
        }
    )
    fig = plot_oracle_coverage(df)
    assert fig is not None
    assert fig.layout.legend.title.text == "n"
    assert fig.layout.xaxis.title.text in (None, "")
    assert fig.layout.yaxis.title.text in (None, "")


def test_plot_oracle_coverage_requires_columns() -> None:
    df = pd.DataFrame({"T": [120], "S_true": [0.0], "coverage": [0.95]})
    try:
        plot_oracle_coverage(df)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing mc_lo/mc_hi.")


def test_plot_oracle_coverage_requires_n_or_t_column() -> None:
    df = pd.DataFrame(
        {
            "S_true": [0.0],
            "coverage": [0.95],
            "mc_lo": [0.90],
            "mc_hi": [0.99],
        }
    )
    try:
        plot_oracle_coverage(df)
    except ValueError as exc:
        assert "expected `n` or `T`" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing n/T column.")


def test_plot_oracle_se_smoke(tmp_path, monkeypatch) -> None:
    calls = _stub_write_image(monkeypatch)
    df = pd.DataFrame(
        {
            "n": [36, 36, 120, 120],
            "S_true": [-0.5, 0.5, -0.5, 0.5],
            "se_cell": [0.20, 0.22, 0.11, 0.12],
        }
    )
    out_html = tmp_path / "oracle-se.html"
    fig = plot_oracle_se(df, out_html=str(out_html))
    assert fig is not None
    assert [p for p, _ in calls] == [out_html.with_suffix(".png")]
    assert out_html.with_suffix(".png").exists()
    assert not out_html.exists()
    assert fig.layout.legend.title.text == "n"
    assert fig.layout.xaxis.title.text in (None, "")
    assert fig.layout.yaxis.title.text in (None, "")
    ann = _annotation_texts(fig)
    assert "True Sharpe" in ann
    assert "Average Oracle SE" in ann


def test_plot_oracle_se_accepts_legacy_t_alias() -> None:
    df = pd.DataFrame(
        {
            "T": [36, 36, 120, 120],
            "S_true": [-0.5, 0.5, -0.5, 0.5],
            "se_cell": [0.20, 0.22, 0.11, 0.12],
        }
    )
    fig = plot_oracle_se(df)
    assert fig is not None
    assert fig.layout.legend.title.text == "n"


def test_plot_oracle_se_requires_metric_column() -> None:
    df = pd.DataFrame({"n": [36], "S_true": [0.0]})
    try:
        plot_oracle_se(df)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing se_cell.")


def test_write_plotly_png_coerces_html_alias_and_preserves_png(monkeypatch, tmp_path) -> None:
    calls = _stub_write_image(monkeypatch)

    out_from_html = write_plotly_png(Figure(), tmp_path / "oracle.html")
    out_from_png = write_plotly_png(Figure(), tmp_path / "oracle-se.png", width=1800, height=1200, scale=3.0)

    assert out_from_html == tmp_path / "oracle.png"
    assert out_from_png == tmp_path / "oracle-se.png"
    assert [p for p, _ in calls] == [tmp_path / "oracle.png", tmp_path / "oracle-se.png"]
    assert calls[0][1]["width"] == 3200
    assert calls[0][1]["height"] == 2000
    assert calls[0][1]["scale"] == 2.0
    assert calls[1][1]["width"] == 1800
    assert calls[1][1]["height"] == 1200
    assert calls[1][1]["scale"] == 3.0


def test_write_plotly_png_raises_clear_error_when_backend_missing(monkeypatch, tmp_path) -> None:
    def fail_write_image(self, path, *args, **kwargs) -> None:
        raise ModuleNotFoundError("No module named 'kaleido'")

    monkeypatch.setattr(BaseFigure, "write_image", fail_write_image)

    try:
        write_plotly_png(Figure(), tmp_path / "oracle.html")
    except RuntimeError as exc:
        assert "Plotly PNG export failed" in str(exc)
        assert "kaleido" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when Plotly PNG export backend is unavailable.")
