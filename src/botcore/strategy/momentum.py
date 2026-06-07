"""Donchian breakout momentum strategy with a trend-regime filter.

Logic (long-only by default; shorts symmetric when enabled):
  * Regime filter: only take longs when price is above a slow SMA AND the fast
    EMA is above the slow EMA. This keeps us out of chop/downtrends, which is
    where naive breakout systems bleed.
  * Entry: close breaks above the highest high of the last `breakout` bars.
  * Exit: close breaks below the lowest low of the last `breakout//2` bars
    (a Donchian trailing exit). Stops/risk sizing are handled by the engine via
    the ATR column.

This is intentionally simple and robust. The edge, if any, comes from crypto's
tendency to trend; the regime filter and trailing exit control the downside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import atr, ema, rolling_high, rolling_low, sma
from .base import Strategy


class MomentumBreakout(Strategy):
    name = "momentum"

    def __init__(
        self,
        fast: int = 24,
        slow: int = 96,
        breakout: int = 48,
        atr_window: int = 14,
        regime_window: int = 200,
        allow_short: bool = False,
        **params,
    ):
        super().__init__(
            fast=fast, slow=slow, breakout=breakout, atr_window=atr_window,
            regime_window=regime_window, allow_short=allow_short, **params,
        )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]

        ema_fast = ema(close, p["fast"])
        ema_slow = ema(close, p["slow"])
        regime = sma(close, p["regime_window"])

        # Use prior bar's channel so the current close breaking it is a real,
        # tradeable event (shift(1) => no peeking at the bar we trade on).
        upper = rolling_high(df["high"], p["breakout"]).shift(1)
        lower_exit = rolling_low(df["low"], max(2, p["breakout"] // 2)).shift(1)
        lower = rolling_low(df["low"], p["breakout"]).shift(1)
        upper_exit = rolling_high(df["high"], max(2, p["breakout"] // 2)).shift(1)

        long_trend = (ema_fast > ema_slow) & (close > regime)
        long_entry = (close > upper) & long_trend
        long_exit = close < lower_exit

        short_trend = (ema_fast < ema_slow) & (close < regime)
        short_entry = (close < lower) & short_trend
        short_exit = close > upper_exit

        signal = self._build_position(
            long_entry, long_exit, short_entry, short_exit,
            allow_short=p["allow_short"],
        )

        out = pd.DataFrame(index=df.index)
        out["signal"] = signal
        out["atr"] = atr(df, p["atr_window"])
        return out

    @staticmethod
    def _build_position(
        long_entry: pd.Series,
        long_exit: pd.Series,
        short_entry: pd.Series,
        short_exit: pd.Series,
        allow_short: bool,
    ) -> pd.Series:
        """Convert entry/exit events into a held position series (state machine).

        Holds +1 from a long entry until a long exit (or short entry if shorts
        are enabled), and symmetrically for shorts. Pure-python loop for clarity;
        it's O(n) and runs in milliseconds for typical bar counts.
        """
        le = long_entry.to_numpy()
        lx = long_exit.to_numpy()
        se = short_entry.to_numpy()
        sx = short_exit.to_numpy()
        n = len(le)
        pos = np.zeros(n, dtype=np.int8)
        state = 0
        for i in range(n):
            if state == 0:
                if le[i]:
                    state = 1
                elif allow_short and se[i]:
                    state = -1
            elif state == 1:
                if lx[i]:
                    state = -1 if (allow_short and se[i]) else 0
            elif state == -1:
                if sx[i]:
                    state = 1 if le[i] else 0
            pos[i] = state
        return pd.Series(pos, index=long_entry.index, name="signal")
