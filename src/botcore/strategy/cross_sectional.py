"""Cross-sectional momentum signal.

Given a panel of daily closes (one column per coin), this computes *target
portfolio weights* on each rebalance date. The logic:

  1. For each coin, measure trailing momentum = return over the last `lookback`
     days, skipping the most recent `skip` days. The skip avoids short-term
     mean-reversion (the last few days often snap back, which is noise here).
  2. Rank coins by that momentum.
  3. Long the top `top_k` coins, equal-weight. (Optionally short the bottom
     `top_k` if `dollar_neutral` — disabled by default; crypto shorts get
     squeezed and we're optimising for Sharpe, not bravado.)

Output: a weight panel (same index/columns as prices) where each row sums to 1.0
on the long-only side. Rows between rebalances repeat the last decision; the
portfolio engine handles the actual drift + trading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_weights(
    closes: pd.DataFrame,
    lookback: int = 30,
    skip: int = 2,
    top_k: int = 5,
    rebalance_days: int = 7,
    dollar_neutral: bool = False,
    regime_ma: int | None = 50,
) -> pd.DataFrame:
    """Compute target weights from cross-sectional momentum.

    closes: DataFrame of daily closes (index=dates, cols=symbols).
    regime_ma: if set, an aggregate index-trend filter. Build an equal-weight
        index of the universe; when it is below its own `regime_ma`-day moving
        average (market in a downtrend), the whole book goes to cash. Set to
        None to disable. An index-trend gate cleanly sidesteps bear regimes; a
        per-coin breadth gate (tested) just whipsaws, so we use index trend.
    Returns a DataFrame of target weights, forward-filled between rebalances and
    masked daily by the regime gate (the engine reads the row valid at each date).
    """
    # Trailing momentum: pct change over `lookback`, ending `skip` days ago.
    # shift(skip) moves the window's end back, so today's value uses data up to
    # `skip` days ago -> strictly causal, no lookahead.
    shifted = closes.shift(skip)
    momentum = shifted / shifted.shift(lookback) - 1.0

    n = len(closes)
    # Start all-NaN; only rebalance rows get an explicit full weight vector.
    # Forward-filling then carries the *entire* vector (zeros included) between
    # rebalances, so a coin that drops out of the top-k correctly goes to 0.
    weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)

    for r in range(0, n, rebalance_days):
        row = momentum.iloc[r]
        valid = row.dropna()
        if len(valid) < top_k:
            continue  # not enough history yet

        ranked = valid.sort_values(ascending=False)
        w = pd.Series(0.0, index=closes.columns)
        w[ranked.index[:top_k]] = 1.0 / top_k

        if dollar_neutral and len(valid) >= 2 * top_k:
            w[ranked.index[-top_k:]] = -1.0 / top_k

        weights.iloc[r] = w.values

    # Carry each rebalance decision forward; zero before the first valid one.
    weights = weights.ffill().fillna(0.0)

    # Aggregate index-trend regime gate: hold cash when the equal-weight index
    # is below its `regime_ma`-day MA. Causal (index uses closes up to day t;
    # the engine lags weights by one day before applying them).
    if regime_ma:
        index = (1.0 + closes.pct_change().mean(axis=1)).cumprod()
        gate = (index > index.rolling(regime_ma).mean()).astype(float)
        weights = weights.mul(gate, axis=0)

    return weights
