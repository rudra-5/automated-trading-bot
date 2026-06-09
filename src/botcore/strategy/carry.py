"""Funding-carry sleeve: delta-neutral long-spot / short-perp basis harvest.

Each rebalance, hold an equal-weight basket of the coins whose recent funding is
positive (longs paying shorts). A long-spot + short-perp pair in each earns that
coin's funding while the price move cancels out. Returns are the funding drip
minus the cost of trading both legs when the basket changes.

We never hold negative-funding coins (we'd be paying), so the sleeve is a clean
"collect the crowd's leverage premium" trade. Output: a daily net return series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def carry_returns(
    funding_daily: pd.DataFrame,
    select_lookback: int = 7,
    rebalance_days: int = 7,
    min_funding: float = 0.0,
    top_k: int | None = None,
    cost_bps: float = 6.0,
    legs: int = 2,
) -> pd.Series:
    """Daily net returns of the carry sleeve.

    funding_daily: daily funding-rate panel (index=date, cols=symbols).
    select_lookback: days of trailing funding used to choose the basket.
    cost_bps: per-side cost; multiplied by `legs` (spot + perp) on turnover.
    """
    # Causal selection signal: trailing-mean funding ending yesterday.
    trail = funding_daily.rolling(select_lookback).mean().shift(1)

    n = len(funding_daily)
    cols = funding_daily.columns
    weights = pd.DataFrame(np.nan, index=funding_daily.index, columns=cols)

    for r in range(0, n, rebalance_days):
        row = trail.iloc[r]
        pos = row[row > min_funding].dropna()
        if pos.empty:
            weights.iloc[r] = 0.0
            continue
        if top_k is not None:
            pos = pos.sort_values(ascending=False).iloc[:top_k]
        w = pd.Series(0.0, index=cols)
        w[pos.index] = 1.0 / len(pos)
        weights.iloc[r] = w.values

    weights = weights.ffill().fillna(0.0)

    # Earn today's funding on yesterday's basket (causal).
    w_held = weights.shift(1).fillna(0.0)
    gross = (w_held * funding_daily.fillna(0.0)).sum(axis=1)

    # Turnover cost when the basket changes: both legs, per side.
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4) * legs

    return (gross - cost).rename("carry")
