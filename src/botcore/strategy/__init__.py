"""Strategy registry."""
from __future__ import annotations

from .base import Strategy
from .mean_reversion import MeanReversion
from .momentum import MomentumBreakout

_REGISTRY = {
    MomentumBreakout.name: MomentumBreakout,
    MeanReversion.name: MeanReversion,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))


__all__ = ["Strategy", "MomentumBreakout", "MeanReversion", "build_strategy"]
