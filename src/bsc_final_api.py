"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.bsc_final_api is deprecated; use finance_data.bsc instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .finance_data.bsc import api as _api

for _name in dir(_api):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_api, _name)

__all__ = [name for name in dir(_api) if not name.startswith("_")]
