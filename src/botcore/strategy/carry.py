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


def _basket_weights(funding_daily, select_lookback, rebalance_days, min_funding, top_k):
    """Shared basket-selection logic (positive trailing-funding coins)."""
    trail = funding_daily.rolling(select_lookback).mean().shift(1)
    n, cols = len(funding_daily), funding_daily.columns
    w = pd.DataFrame(np.nan, index=funding_daily.index, columns=cols)
    for r in range(0, n, rebalance_days):
        pos = trail.iloc[r]
        pos = pos[pos > min_funding].dropna()
        if pos.empty:
            w.iloc[r] = 0.0
            continue
        if top_k is not None:
            pos = pos.sort_values(ascending=False).iloc[:top_k]
        row = pd.Series(0.0, index=cols)
        row[pos.index] = 1.0 / len(pos)
        w.iloc[r] = row.values
    return w.ffill().fillna(0.0)


def carry_returns_honest(
    funding_daily: pd.DataFrame,
    spot_closes: pd.DataFrame,
    perp_closes: pd.DataFrame,
    select_lookback: int = 7,
    rebalance_days: int = 7,
    min_funding: float = 0.0,
    top_k: int | None = None,
    cost_bps: float = 6.0,
    legs: int = 2,
    margin_fraction: float = 0.30,
) -> pd.Series:
    """Carry returns with basis risk and a capital-efficiency haircut.

    Per unit notional, daily PnL = funding + (spot_ret - perp_ret). The basis
    term is the imperfect-hedge tracking error: ~0 in calm markets, sharply
    negative when the perp spikes above spot (short squeeze) — the real risk the
    naive model ignored. Return is then taken on *deployed* capital, which is
    notional * (1 + margin_fraction) because the short perp locks margin.
    """
    idx = spot_closes.index
    funding = funding_daily.reindex(idx).fillna(0.0)
    spot_ret = spot_closes.pct_change()
    perp_ret = perp_closes.reindex(idx).pct_change()

    # Per-coin carry PnL on notional (long spot, short perp, collect funding).
    coin_pnl = (funding + (spot_ret - perp_ret)).fillna(0.0)

    cols = [c for c in funding_daily.columns if c in spot_closes.columns and c in perp_closes.columns]
    weights = _basket_weights(funding_daily[cols], select_lookback, rebalance_days, min_funding, top_k)
    weights = weights.reindex(idx).ffill().fillna(0.0)

    w_held = weights.shift(1).fillna(0.0)
    gross = (w_held * coin_pnl[cols]).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4) * legs

    net_on_notional = gross - cost
    return (net_on_notional / (1.0 + margin_fraction)).rename("carry_honest")
