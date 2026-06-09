#!/usr/bin/env python3
"""Phase B: blend cross-sectional momentum (directional) with funding carry
(market-neutral) into one vol-targeted portfolio.

The pitch: two positive-return streams that don't move together. Momentum makes
money when the cross-section trends; carry makes money from the perpetual
funding premium regardless of direction. Blending them keeps the return while
cutting the volatility -> higher Sharpe than either sleeve alone.

Each sleeve is scaled to a common vol target, then equal-weighted. We report
each sleeve standalone and the blend, full-sample and out-of-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from botcore.backtest.portfolio import PortfolioConfig, run_portfolio
from botcore.data.funding import load_funding
from botcore.data.universe import load_universe
from botcore.metrics.performance import compute_metrics
from botcore.strategy.carry import carry_returns
from botcore.strategy.cross_sectional import momentum_weights

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT",
    "LTC/USDT", "LINK/USDT", "BCH/USDT", "XLM/USDT", "TRX/USDT", "ETC/USDT",
    "ATOM/USDT", "VET/USDT", "SOL/USDT", "DOT/USDT", "UNI/USDT", "XTZ/USDT",
    "ALGO/USDT", "FIL/USDT",
]
PPY = 365


def vol_target(rets: pd.Series, target: float, lookback: int = 30,
               max_lev: float = 4.0, ppy: int = PPY) -> pd.Series:
    """Scale a return stream to a constant vol target (causal, lagged)."""
    daily_target = target / np.sqrt(ppy)
    roll = rets.rolling(lookback).std().shift(1)
    lev = (daily_target / roll).replace([np.inf, -np.inf], np.nan).clip(upper=max_lev).fillna(0.0)
    return lev * rets


def equity_of(rets: pd.Series, initial: float = 10_000.0) -> pd.Series:
    return initial * (1.0 + rets.fillna(0.0)).cumprod()


def report(label: str, equity: pd.Series, ppy: int = PPY) -> None:
    rets = equity.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=equity.index)
    m = compute_metrics(equity, rets, pos, pd.DataFrame(), ppy)
    print(f"\n=== {label} ===")
    print(f"  Final equity   : {m.final_equity:,.0f}")
    print(f"  Total return   : {m.total_return:+.2%}")
    print(f"  CAGR           : {m.cagr:+.2%}")
    print(f"  Ann. vol       : {m.ann_volatility:.2%}")
    print(f"  Sharpe         : {m.sharpe:.2f}")
    print(f"  Sortino        : {m.sortino:.2f}")
    print(f"  Max drawdown   : {m.max_drawdown:.2%}")
    print(f"  Calmar         : {m.calmar:.2f}")


def main() -> None:
    closes = load_universe(UNIVERSE)
    funding = load_funding(UNIVERSE).reindex(closes.index).fillna(0.0)
    print(f"Universe: {len(closes.columns)} coins, {len(closes)} daily bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    SLEEVE_VOL = 0.15  # each sleeve scaled to 15% vol before blending

    # --- Sleeve A: cross-sectional momentum ---
    weights = momentum_weights(closes, lookback=30, skip=2, top_k=5, rebalance_days=7)
    mom = run_portfolio(
        closes, weights,
        PortfolioConfig(cost_bps=9.5, target_vol=SLEEVE_VOL, max_leverage=3.0, periods_per_year=PPY),
    )
    mom_ret = mom.returns

    # --- Sleeve B: funding carry ---
    carry_raw = carry_returns(funding, select_lookback=7, rebalance_days=7, cost_bps=6.0)
    carry_ret = vol_target(carry_raw, SLEEVE_VOL)  # lever the low-vol carry up to 15%

    # --- Blend: equal risk weight, then cap combined vol at 15% ---
    common = mom_ret.index.intersection(carry_ret.index)
    blend_raw = 0.5 * mom_ret.reindex(common).fillna(0.0) + 0.5 * carry_ret.reindex(common).fillna(0.0)
    blend_ret = vol_target(blend_raw, SLEEVE_VOL)

    corr = mom_ret.reindex(common).fillna(0.0).corr(carry_ret.reindex(common).fillna(0.0))
    print(f"Sleeve correlation (mom vs carry): {corr:+.2f}   "
          f"[low/negative = good diversification]")

    mom_eq = equity_of(mom_ret)
    carry_eq = equity_of(carry_ret)
    blend_eq = equity_of(blend_ret)
    btc_eq = equity_of(closes["BTC/USDT"].pct_change())

    for lbl, eq in [("MOMENTUM only", mom_eq), ("CARRY only", carry_eq),
                    ("BLEND 50/50", blend_eq), ("BENCH: BTC hold", btc_eq)]:
        report(f"{lbl} (full)", eq)

    split = int(len(closes) * 0.7)
    cut = closes.index[split]
    print(f"\n----- OUT-OF-SAMPLE (from {cut.date()}) -----")
    for lbl, eq in [("MOMENTUM only", mom_eq), ("CARRY only", carry_eq),
                    ("BLEND 50/50", blend_eq), ("BENCH: BTC hold", btc_eq)]:
        report(f"{lbl} (OOS)", eq.loc[cut:])


if __name__ == "__main__":
    main()
