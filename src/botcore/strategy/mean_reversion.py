"""Z-score mean-reversion strategy (counter-trend).

Fades stretched moves: go long when price is `entry_z` standard deviations below
its rolling mean, exit when it reverts past `exit_z`. Symmetric for shorts when
enabled. Optionally gated by a trend filter so we don't fade a strong trend
(the classic way mean-reversion blows up).

Included mainly as a contrast to the momentum strategy and to exercise the
short side of the engine. On trending crypto majors it typically underperforms
momentum unless restricted to ranging regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import atr, ema, zscore
from .base import Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        window: int = 48,
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        atr_window: int = 14,
        trend_filter: int = 200,
        allow_short: bool = True,
        **params,
    ):
        super().__init__(
            window=window, entry_z=entry_z, exit_z=exit_z, atr_window=atr_window,
            trend_filter=trend_filter, allow_short=allow_short, **params,
        )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        z = zscore(close, p["window"])

        # Trend filter: only fade in the direction that is not strongly trending.
        trend = ema(close, p["trend_filter"])
        below_trend = close < trend
        above_trend = close > trend

        n = len(df)
        zv = z.to_numpy()
        long_ok = (~above_trend).to_numpy()  # don't long fade in a strong uptrend... allow when not above
        short_ok = (~below_trend).to_numpy()

        pos = np.zeros(n, dtype=np.int8)
        state = 0
        entry_z, exit_z = p["entry_z"], p["exit_z"]
        allow_short = p["allow_short"]
        for i in range(n):
            zi = zv[i]
            if np.isnan(zi):
                pos[i] = state
                continue
            if state == 0:
                if zi <= -entry_z and long_ok[i]:
                    state = 1
                elif allow_short and zi >= entry_z and short_ok[i]:
                    state = -1
            elif state == 1:
                if zi >= -exit_z:
                    state = 0
            elif state == -1:
                if zi <= exit_z:
                    state = 0
            pos[i] = state

        out = pd.DataFrame(index=df.index)
        out["signal"] = pd.Series(pos, index=df.index)
        out["atr"] = atr(df, p["atr_window"])
        return out
