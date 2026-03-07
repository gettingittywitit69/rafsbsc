"""Notebook-facing finance data package.

Submodules are loaded lazily to avoid importing optional dependencies
when users only need a subset of the package (for example, `finance_data.bsc`).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["ar_garch", "bsc", "datasets", "french", "metrics", "spreads", "survival"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
