#!/usr/bin/env python3
"""Bear-regime stress test.

Runs the same momentum + carry + blend pipeline, then slices the equity curves
to specific bear windows and reports how each sleeve held up versus simply
holding BTC. The question: does the edge survive when crypto is falling, or was
it just riding a bull market?

Windows (all inside the 2021-2024 data, all 20 coins live):
  * 2022 full year   : BTC ~-65%, the grinding bear
  * LUNA crash       : May-Jun 2022, ~$40B algo-stablecoin implosion
  * FTX crash        : Nov 2022, #2 exchange collapse, forced deleveraging
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from botcore.backtest.portfolio import PortfolioConfig, run_portfolio
from botcore.data.funding import load_funding, load_perp_closes
from botcore.data.universe import load_universe
from botcore.metrics.performance import compute_metrics
from botcore.strategy.carry import carry_returns_honest
from botcore.strategy.cross_sectional import momentum_weights

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT",
    "LTC/USDT", "LINK/USDT", "BCH/USDT", "XLM/USDT", "TRX/USDT", "ETC/USDT",
    "ATOM/USDT", "VET/USDT", "SOL/USDT", "DOT/USDT", "UNI/USDT", "XTZ/USDT",
    "ALGO/USDT", "FIL/USDT",
]
PPY = 365
SLEEVE_VOL = 0.15

WINDOWS = {
    "2022 FULL BEAR (Jan-Dec 2022)": ("2022-01-01", "2022-12-31"),
    "LUNA CRASH (May-Jun 2022)":     ("2022-05-01", "2022-06-30"),
    "FTX CRASH (Nov 2022)":          ("2022-11-01", "2022-11-30"),
}


def vol_target(rets, target, lookback=30, max_lev=4.0, ppy=PPY):
    daily_target = target / np.sqrt(ppy)
    roll = rets.rolling(lookback).std().shift(1)
    lev = (daily_target / roll).replace([np.inf, -np.inf], np.nan).clip(upper=max_lev).fillna(0.0)
    return lev * rets


def equity_of(rets, initial=10_000.0):
    return initial * (1.0 + rets.fillna(0.0)).cumprod()


def window_stats(equity: pd.Series, lo: str, hi: str) -> dict:
    sl = equity.loc[lo:hi]
    if len(sl) < 2:
        return {}
    rets = sl.pct_change().fillna(0.0)
    ret = sl.iloc[-1] / sl.iloc[0] - 1.0
    dd = float((sl / sl.cummax() - 1.0).min())
    vol = float(rets.std(ddof=0) * np.sqrt(PPY))
    sharpe = float(rets.mean() * PPY / vol) if vol > 0 else 0.0
    return {"ret": ret, "dd": dd, "sharpe": sharpe}


def main() -> None:
    closes = load_universe(UNIVERSE)
    funding = load_funding(UNIVERSE).reindex(closes.index).fillna(0.0)
    perp = load_perp_closes(UNIVERSE).reindex(closes.index)

    weights = momentum_weights(closes, lookback=30, skip=2, top_k=5, rebalance_days=7,
                              weighting="invvol")
    mom = run_portfolio(
        closes, weights,
        PortfolioConfig(cost_bps=9.5, target_vol=SLEEVE_VOL, max_leverage=3.0, periods_per_year=PPY),
    )
    carry_raw = carry_returns_honest(funding, closes, perp, 7, 7, cost_bps=6.0, margin_fraction=0.30)
    carry = vol_target(carry_raw, SLEEVE_VOL, max_lev=2.0)
    common = mom.returns.index.intersection(carry.index)
    # Equal-risk blend of two already-vol-targeted sleeves; no second re-lever.
    blend = 0.5 * mom.returns.reindex(common).fillna(0.0) + 0.5 * carry.reindex(common).fillna(0.0)

    curves = {
        "Momentum": equity_of(mom.returns),
        "Carry":    equity_of(carry),
        "Blend":    equity_of(blend),
        "BTC hold": equity_of(closes["BTC/USDT"].pct_change()),
    }

    for title, (lo, hi) in WINDOWS.items():
        print(f"\n================ {title} ================")
        print(f"{'Strategy':<10} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8}")
        for name, eq in curves.items():
            s = window_stats(eq, lo, hi)
            if not s:
                continue
            print(f"{name:<10} {s['ret']:>+8.1%} {s['dd']:>+8.1%} {s['sharpe']:>8.2f}")


if __name__ == "__main__":
    main()
