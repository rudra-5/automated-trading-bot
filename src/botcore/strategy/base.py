"""Strategy interface.

A strategy turns OHLCV into a *target direction* per bar. It must use only
information available up to and including that bar's close — the backtest engine
executes a bar's signal at the NEXT bar's open, so any lookahead here becomes
fake profit. Keep indicator windows causal (no centered/forward-filled values).

generate() returns a DataFrame aligned to the input index with columns:
  - signal: target direction in {-1, 0, +1} (engine handles sizing)
  - atr:    ATR value used by the risk module for stop distance / sizing
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
