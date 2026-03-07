from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_run_partA_min_module():
    module_path = Path(__file__).resolve().parents[1] / "notebooks" / "run_partA_min.py"
    spec = spec_from_file_location("run_partA_min_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyFigure:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, dict[str, object]]] = []

    def savefig(self, path, **kwargs) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        self.calls.append((out_path, kwargs))


def test_run_partA_min_saves_png_only(tmp_path) -> None:
    module = _load_run_partA_min_module()
    fig = _DummyFigure()

    module.savefig(fig, tmp_path / "partA_coverage_95")

    assert fig.calls == [
        (
            tmp_path / "partA_coverage_95.png",
            {"dpi": 160, "bbox_inches": "tight"},
        )
    ]
    assert (tmp_path / "partA_coverage_95.png").exists()
    assert not (tmp_path / "partA_coverage_95.pdf").exists()
