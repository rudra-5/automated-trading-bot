"""Portfolio backtest engine for multi-asset, weight-based strategies.

Unlike the single-asset engine (which holds one position and an ATR stop), this
one holds a *vector* of dollar positions across many coins and rebalances toward
target weights on a schedule. It models the two things that decide whether a
cross-sectional strategy is real:

  * Turnover cost: each rebalance trades from current (drifted) holdings to the
    new target. Cost is charged on the dollar turnover, per side. Low turnover
    (weekly) is the whole reason this can work where the 1h strategies bled.
  * Volatility targeting: exposure is scaled so the portfolio runs near a target
    annual volatility. Computed causally from the *unlevered* portfolio's
    trailing realised vol, then lagged one day — no peeking at today's risk.

No lookahead: target weights decided as of day t-1 are applied to day t returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    cost_bps: float = 9.5          # per-side fee+slippage on turnover (bps)
    target_vol: float = 0.20       # annualised vol target; None-like 0 disables
    vol_lookback: int = 30         # days of returns for the vol estimate
    max_leverage: float = 2.0      # cap on gross exposure
    periods_per_year: int = 365    # daily bars


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    gross_exposure: pd.Series      # leverage actually applied each day
    weights: pd.DataFrame          # realised (drifted) weights per day
    turnover: pd.Series            # traded fraction of equity per day


def _leverage_series(unlevered_ret: pd.Series, cfg: PortfolioConfig) -> pd.Series:
    """Scale factor to hit target vol, from trailing realised vol (lagged)."""
    if not cfg.target_vol or cfg.target_vol <= 0:
        return pd.Series(1.0, index=unlevered_ret.index)
    daily_target = cfg.target_vol / np.sqrt(cfg.periods_per_year)
    roll_vol = unlevered_ret.rolling(cfg.vol_lookback).std().shift(1)
    lev = daily_target / roll_vol
    lev = lev.replace([np.inf, -np.inf], np.nan).clip(upper=cfg.max_leverage)
    return lev.fillna(1.0)


def run_portfolio(
    closes: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: PortfolioConfig,
    initial_capital: float = 10_000.0,
) -> PortfolioResult:
    """Simulate a weight-based portfolio with turnover costs and vol targeting."""
    closes = closes.sort_index()
    rets = closes.pct_change().fillna(0.0)

    # Decision made on day t-1 is applied to day t (no lookahead).
    w_target = target_weights.reindex(closes.index).shift(1).fillna(0.0)

    # Pass 1: unlevered portfolio returns -> trailing vol -> leverage.
    unlevered = (w_target * rets).sum(axis=1)
    leverage = _leverage_series(unlevered, cfg)

    cost_rate = cfg.cost_bps / 1e4
    syms = closes.columns
    r = rets.to_numpy()
    wt = w_target.to_numpy()
    lev = leverage.to_numpy()
    n = len(closes)

    cash = initial_capital
    pos = np.zeros(len(syms))           # signed dollars per asset
    equity_curve = np.empty(n)
    gross = np.zeros(n)
    turnover = np.zeros(n)
    realised_w = np.zeros((n, len(syms)))

    prev_target = np.full(len(syms), np.nan)
    for t in range(n):
        # 1) Mark holdings to today's returns (cash unaffected).
        pos = pos * (1.0 + r[t])
        equity = cash + pos.sum()

        # 2) Rebalance only when the target vector changes (weekly here).
        target_row = wt[t]
        if equity > 0 and not np.allclose(target_row, prev_target, equal_nan=True):
            desired = lev[t] * target_row * equity     # dollars per asset
            trades = desired - pos
            traded_notional = np.abs(trades).sum()
            cost = traded_notional * cost_rate
            cash -= trades.sum() + cost
            pos = desired
            equity = cash + pos.sum()
            turnover[t] = traded_notional / initial_capital if initial_capital else 0.0
            prev_target = target_row.copy()

        equity_curve[t] = equity
        gross[t] = np.abs(pos).sum() / equity if equity > 0 else 0.0
        realised_w[t] = pos / equity if equity > 0 else 0.0

    idx = closes.index
    equity = pd.Series(equity_curve, index=idx, name="equity")
    return PortfolioResult(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        gross_exposure=pd.Series(gross, index=idx, name="gross"),
        weights=pd.DataFrame(realised_w, index=idx, columns=syms),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
    )
