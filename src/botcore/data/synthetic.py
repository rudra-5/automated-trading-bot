"""Synthetic OHLCV generator.

Produces a regime-switching geometric-Brownian-motion price path with
volatility clustering, then derives plausible OHLC bars from it. This lets the
whole pipeline (strategies, backtester, tests) run with zero network access and
gives strategies realistic-looking trends and ranges to chew on.

It is NOT a substitute for real market data when evaluating an edge — it has no
microstructure, no fat tails to speak of, and you chose its drift. Use it for
plumbing/CI, use real data (via the ccxt loader) for decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    bars: int = 8760,
    start_price: float = 30_000.0,
    annual_drift: float = 0.20,
    annual_vol: float = 0.70,
    periods_per_year: int = 8760,  # hourly
    seed: int | None = 42,
    start: str = "2021-01-01",
    timeframe_pandas: str = "1h",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dt = 1.0 / periods_per_year
    mu = annual_drift
    base_sigma = annual_vol

    # Regime-switching volatility: occasionally flip between calm and stormy.
    vol_mult = np.ones(bars)
    state = 1.0
    for i in range(bars):
        if rng.random() < 0.01:  # ~1% chance per bar to switch regime
            state = rng.choice([0.6, 1.0, 1.8])
        vol_mult[i] = state
    sigma = base_sigma * vol_mult

    shocks = rng.standard_normal(bars)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(bars)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Intrabar wick noise scaled by local volatility.
    wick = np.abs(rng.standard_normal(bars)) * sigma * np.sqrt(dt) * close
    high = np.maximum(open_, close) + wick * 0.5
    low = np.minimum(open_, close) - wick * 0.5
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=bars) * (1 + vol_mult)

    index = pd.date_range(start=start, periods=bars, freq=timeframe_pandas, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
