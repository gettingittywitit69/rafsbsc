from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from plotly.graph_objs import Figure


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


def _resolve_size_column(df: pd.DataFrame) -> str:
    if "n" in df.columns:
        return "n"
    if "T" in df.columns:
        return "T"
    raise ValueError("Missing required sample-size column: expected `n` or `T`.")


def _prepare_oracle_line_df(df: pd.DataFrame, *, required: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    size_col = _resolve_size_column(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_df = df.copy()
    if size_col == "T":
        plot_df["n"] = plot_df["T"]

    n_numeric = pd.to_numeric(plot_df["n"], errors="coerce")
    if n_numeric.isna().any():
        raise ValueError("Sample-size column `n`/`T` must be numeric.")
    plot_df["n"] = n_numeric.astype(int)
    plot_df["n_str"] = plot_df["n"].astype(str)
    return plot_df


def _resolve_plotly_png_path(out_html: str | Path) -> Path:
    out_path = Path(out_html)
    if out_path.suffix.lower() == ".png":
        return out_path
    if out_path.suffix:
        return out_path.with_suffix(".png")
    return out_path.with_suffix(".png")


def write_plotly_png(
    fig: Any,
    out_html: str | Path,
    *,
    width: int = 3200,
    height: int = 2000,
    scale: float = 2.0,
) -> Path:
    out_path = _resolve_plotly_png_path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(
            str(out_path),
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Plotly PNG export failed for `{out_path}`. Ensure `kaleido` is installed and a Chrome/Chromium binary is available."
        ) from exc
    return out_path


def _write_plotly_artifacts(
    fig: Any,
    out_html: str | None,
    *,
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> None:
    if out_html is not None:
        write_plotly_png(fig, out_html, width=png_width, height=png_height, scale=png_scale)

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
    except Exception:
        pass


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


def _plot_oracle_line_figure(
    plot_df: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    out_html: str | None,
    y_range: tuple[float, float] | None,
    nominal: float | None = None,
    use_error_bars: bool = False,
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> Any:
    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Oracle Plotly figures require plotly. Install it with `pip install plotly`.") from exc

    n_order = [str(v) for v in sorted(plot_df["n"].dropna().astype(int).unique())]
    line_kwargs: dict[str, Any] = {"category_orders": {"n_str": n_order}}
    if use_error_bars:
        line_kwargs["error_y"] = "err_y"
        line_kwargs["error_y_minus"] = "err_y_minus"

    frame = plot_df.sort_values([c for c in ("dgp", "n", "S_true") if c in plot_df.columns]).reset_index(drop=True)
    if "dgp" in frame.columns:
        fig = px.line(
            frame,
            x="S_true",
            y=y,
            color="n_str",
            facet_col="dgp",
            markers=True,
            **line_kwargs,
        )
        if nominal is not None:
            fig.add_hline(y=float(nominal), row="all", col="all")
    else:
        fig = px.line(
            frame,
            x="S_true",
            y=y,
            color="n_str",
            markers=True,
            **line_kwargs,
        )
        if nominal is not None:
            fig.add_hline(y=float(nominal))

    _install_plotly_ipython_fallback(fig)
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("dgp=", "")))
    if y_range is None:
        pass
    else:
        if len(y_range) != 2:
            raise ValueError("y_range must be a (low, high) tuple.")
        fig.update_yaxes(range=list(y_range))
    fig.update_layout(legend_title_text="n", margin=dict(l=50, r=20, t=20, b=50))
    _apply_shared_axis_labels(fig, x_label="True Sharpe", y_label=ylabel)
    _write_plotly_artifacts(fig, out_html, png_width=png_width, png_height=png_height, png_scale=png_scale)
    return fig


def plot_oracle_coverage(
    df: pd.DataFrame,
    *,
    out_html: str | None = None,
    nominal: float = 0.95,
    y_range: tuple[float, float] = (0.85, 1.0),
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> Any:
    """
    Plot oracle-analytic Monte Carlo coverage with 95% MC error bars.

    Accepts `n` as the primary sample-size column and `T` as a backward-compatible alias.
    If `out_html` is provided, the figure is exported as PNG; `.html` is accepted as a
    backward-compatible basename alias.
    """
    plot_df = _prepare_oracle_line_df(df, required=("S_true", "coverage", "mc_lo", "mc_hi"))
    plot_df["err_y"] = (plot_df["mc_hi"] - plot_df["coverage"]).clip(lower=0.0)
    plot_df["err_y_minus"] = (plot_df["coverage"] - plot_df["mc_lo"]).clip(lower=0.0)
    return _plot_oracle_line_figure(
        plot_df,
        y="coverage",
        ylabel="95% Coverage",
        out_html=out_html,
        y_range=y_range,
        nominal=nominal,
        use_error_bars=True,
        png_width=png_width,
        png_height=png_height,
        png_scale=png_scale,
    )


def plot_oracle_se(
    df: pd.DataFrame,
    *,
    out_html: str | None = None,
    y_range: tuple[float, float] | None = None,
    png_width: int = 3200,
    png_height: int = 2000,
    png_scale: float = 2.0,
) -> Any:
    """
    Plot average oracle standard errors against the true Sharpe ratio.

    Accepts `n` as the primary sample-size column and `T` as a backward-compatible alias.
    If `out_html` is provided, the figure is exported as PNG; `.html` is accepted as a
    backward-compatible basename alias.
    """
    plot_df = _prepare_oracle_line_df(df, required=("S_true", "se_cell"))
    return _plot_oracle_line_figure(
        plot_df,
        y="se_cell",
        ylabel="Average Oracle SE",
        out_html=out_html,
        y_range=y_range,
        png_width=png_width,
        png_height=png_height,
        png_scale=png_scale,
    )


def smoke_test_plot_oracle_coverage() -> Any:
    dummy = pd.DataFrame(
        {
            "T": [120, 120, 120, 240, 240, 240],
            "S_true": [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5],
            "coverage": [0.93, 0.95, 0.94, 0.95, 0.96, 0.95],
            "mc_lo": [0.90, 0.92, 0.91, 0.92, 0.93, 0.92],
            "mc_hi": [0.96, 0.98, 0.97, 0.98, 0.99, 0.98],
        }
    )
    return plot_oracle_coverage(dummy, out_html="/tmp/test.html")


if __name__ == "__main__":
    smoke_test_plot_oracle_coverage()
