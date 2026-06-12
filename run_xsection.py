#!/usr/bin/env python3
"""Cross-sectional momentum backtest on a crypto universe (Phase A).

Builds a weekly-rebalanced, vol-targeted long-only portfolio of the strongest
recent performers, and compares it to two benchmarks:
  * BTC buy-and-hold     (the obvious "just hold the king" alternative)
  * Equal-weight basket  (hold all coins; isolates whether *selecting* winners
                          beats merely diversifying)

In/out-of-sample split is reported: the OOS block is the honest test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from botcore.backtest.portfolio import PortfolioConfig, run_portfolio
from botcore.data.universe import load_universe
from botcore.metrics.performance import compute_metrics
from botcore.strategy.cross_sectional import momentum_weights

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT",
    "LTC/USDT", "LINK/USDT", "BCH/USDT", "XLM/USDT", "TRX/USDT", "ETC/USDT",
    "ATOM/USDT", "VET/USDT", "SOL/USDT", "DOT/USDT", "UNI/USDT", "XTZ/USDT",
    "ALGO/USDT", "FIL/USDT",
]
PPY = 365


def _equity_from_returns(rets: pd.Series, initial: float = 10_000.0) -> pd.Series:
    return initial * (1.0 + rets.fillna(0.0)).cumprod()


def _report(label: str, equity: pd.Series, ppy: int = PPY) -> None:
    rets = equity.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=equity.index)  # always invested (benchmarks/strategy)
    m = compute_metrics(equity, rets, pos, pd.DataFrame(), ppy)
    print(f"\n=== {label} ===")
    print(f"  Final equity     : {m.final_equity:,.0f}")
    print(f"  Total return     : {m.total_return:+.2%}")
    print(f"  CAGR             : {m.cagr:+.2%}")
    print(f"  Ann. volatility  : {m.ann_volatility:.2%}")
    print(f"  Sharpe           : {m.sharpe:.2f}")
    print(f"  Sortino          : {m.sortino:.2f}")
    print(f"  Max drawdown     : {m.max_drawdown:.2%}")
    print(f"  Calmar           : {m.calmar:.2f}")


def main() -> None:
    closes = load_universe(UNIVERSE)
    print(f"Universe : {len(closes.columns)} coins, {len(closes)} daily bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    weights = momentum_weights(
        closes, lookback=30, skip=2, top_k=5, rebalance_days=7, dollar_neutral=False,
        weighting="invvol",
    )
    cfg = PortfolioConfig(
        cost_bps=9.5, target_vol=0.20, vol_lookback=30, max_leverage=2.0,
        periods_per_year=PPY,
    )
    res = run_portfolio(closes, weights, cfg)

    print(f"Strategy : XS-momentum top5, weekly, vol-target {cfg.target_vol:.0%}, "
          f"max lev {cfg.max_leverage}x, cost {cfg.cost_bps}bps/side")
    print(f"Avg gross exposure: {res.gross_exposure.mean():.2f}x   "
          f"Total turnover: {res.turnover.sum():.1f}x equity")

    # Benchmarks.
    rets = closes.pct_change()
    btc_hold = _equity_from_returns(rets["BTC/USDT"])
    eq_basket = _equity_from_returns(rets.mean(axis=1))

    _report("STRATEGY (full)", res.equity)
    _report("BENCH: BTC hold (full)", btc_hold)
    _report("BENCH: equal-weight basket (full)", eq_basket)

    # Out-of-sample: last 30% is the holdout the rules were not tuned on.
    split = int(len(closes) * 0.7)
    cut = closes.index[split]
    print(f"\n----- OUT-OF-SAMPLE (from {cut.date()}) -----")
    _report("STRATEGY (OOS)", res.equity.loc[cut:])
    _report("BENCH: BTC hold (OOS)", btc_hold.loc[cut:])
    _report("BENCH: equal-weight basket (OOS)", eq_basket.loc[cut:])


if __name__ == "__main__":
    main()
