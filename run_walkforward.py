#!/usr/bin/env python3
"""Walk-forward test of the momentum sleeve (and the blend).

A single in-sample/out-of-sample split can get lucky. Walk-forward is stricter:
roll a training window, pick the best parameters on it, then trade the NEXT
unseen window with those params. Step forward and repeat. Stitching the test
windows gives one continuous equity curve that the parameters never saw during
selection -- the honest test of whether the *tuning process* generalises.

We report:
  * the stitched walk-forward curve (re-optimised each fold)
  * the same periods traded with FIXED default params (did tuning help or hurt?)
  * BTC hold over the same span
  * a per-fold table of chosen params + train/test Sharpe (stability check:
    params lurching every fold = overfit; stable params = a real signal)

Tuning objective: train-window Sharpe of the momentum sleeve. Carry is
parameter-light, so it stays fixed (honest model, 2x cap) and is blended in.
"""
from __future__ import annotations

import sys
from itertools import product
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

# Momentum parameter grid searched on each training window.
GRID_LOOKBACK = [20, 30, 45, 60]
GRID_TOP_K = [3, 5, 7]
GRID_REGIME = [None, 50, 100]

TRAIN_DAYS = 365
TEST_DAYS = 90
STEP_DAYS = 90
DEFAULT_PARAMS = {"lookback": 30, "top_k": 5, "regime_ma": 50}


def vol_target(rets, target, lookback=30, max_lev=4.0, ppy=PPY):
    daily_target = target / np.sqrt(ppy)
    roll = rets.rolling(lookback).std().shift(1)
    lev = (daily_target / roll).replace([np.inf, -np.inf], np.nan).clip(upper=max_lev).fillna(0.0)
    return lev * rets


def sharpe(rets, ppy=PPY):
    r = rets.fillna(0.0)
    vol = r.std(ddof=0) * np.sqrt(ppy)
    return float(r.mean() * ppy / vol) if vol > 0 else 0.0


def equity_of(rets, initial=10_000.0):
    return initial * (1.0 + rets.fillna(0.0)).cumprod()


def report(label, equity, ppy=PPY):
    rets = equity.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=equity.index)
    m = compute_metrics(equity, rets, pos, pd.DataFrame(), ppy)
    print(f"\n=== {label} ===")
    print(f"  Total return : {m.total_return:+.2%}")
    print(f"  CAGR         : {m.cagr:+.2%}")
    print(f"  Ann. vol     : {m.ann_volatility:.2%}")
    print(f"  Sharpe       : {m.sharpe:.2f}")
    print(f"  Sortino      : {m.sortino:.2f}")
    print(f"  Max drawdown : {m.max_drawdown:.2%}")
    print(f"  Calmar       : {m.calmar:.2f}")


def momentum_returns(closes, cfg_port, lookback, top_k, regime_ma):
    """Full-history momentum sleeve daily returns for one parameter set."""
    w = momentum_weights(closes, lookback=lookback, skip=2, top_k=top_k,
                         rebalance_days=7, regime_ma=regime_ma)
    return run_portfolio(closes, w, cfg_port).returns


def main() -> None:
    closes = load_universe(UNIVERSE)
    funding = load_funding(UNIVERSE).reindex(closes.index).fillna(0.0)
    perp = load_perp_closes(UNIVERSE).reindex(closes.index)
    idx = closes.index
    n = len(idx)
    cfg_port = PortfolioConfig(cost_bps=9.5, target_vol=SLEEVE_VOL, max_leverage=3.0, periods_per_year=PPY)

    # Precompute each grid combo's full-history returns once (then slice per fold).
    combos = list(product(GRID_LOOKBACK, GRID_TOP_K, GRID_REGIME))
    print(f"Grid: {len(combos)} momentum param sets; precomputing...")
    combo_ret = {c: momentum_returns(closes, cfg_port, *c) for c in combos}
    default_ret = momentum_returns(closes, cfg_port, DEFAULT_PARAMS["lookback"],
                                   DEFAULT_PARAMS["top_k"], DEFAULT_PARAMS["regime_ma"])

    # Carry sleeve: fixed honest model, blended in.
    carry = vol_target(
        carry_returns_honest(funding, closes, perp, 7, 7, cost_bps=6.0, margin_fraction=0.30),
        SLEEVE_VOL, max_lev=2.0,
    )

    # Build folds and pick best params per training window.
    wf_mom = pd.Series(0.0, index=idx)   # stitched walk-forward momentum returns
    fixed_mom = pd.Series(0.0, index=idx)
    covered = pd.Series(False, index=idx)

    print(f"\n{'Fold':<5}{'Train':<25}{'Test':<25}{'Best params':<28}{'trainSh':>8}{'testSh':>8}")
    k = 0
    while True:
        tr0 = k * STEP_DAYS
        tr1 = tr0 + TRAIN_DAYS
        te1 = tr1 + TEST_DAYS
        if te1 > n:
            break
        train_sl = slice(tr0, tr1)
        test_sl = slice(tr1, te1)

        # Pick combo with best train-window Sharpe.
        best_c, best_s = None, -np.inf
        for c in combos:
            s = sharpe(combo_ret[c].iloc[train_sl])
            if s > best_s:
                best_s, best_c = s, c
        test_ret = combo_ret[best_c].iloc[test_sl]
        test_sh = sharpe(test_ret)

        wf_mom.iloc[test_sl] = test_ret.values
        fixed_mom.iloc[test_sl] = default_ret.iloc[test_sl].values
        covered.iloc[test_sl] = True

        lab = f"L{best_c[0]} K{best_c[1]} R{best_c[2]}"
        print(f"{k:<5}{str(idx[tr0].date())+'..'+str(idx[tr1-1].date()):<25}"
              f"{str(idx[tr1].date())+'..'+str(idx[te1-1].date()):<25}{lab:<28}{best_s:>8.2f}{test_sh:>8.2f}")
        k += 1

    span = covered[covered].index
    lo, hi = span[0], span[-1]
    print(f"\nWalk-forward OOS span: {lo.date()} -> {hi.date()} ({len(span)} days, {k} folds)")

    # Stitch curves over the covered span; blend uses walk-forward momentum + carry.
    wf_mom_s = wf_mom.loc[lo:hi]
    fixed_mom_s = fixed_mom.loc[lo:hi]
    carry_s = carry.loc[lo:hi]
    blend_wf = 0.5 * wf_mom_s + 0.5 * carry_s
    blend_fixed = 0.5 * fixed_mom_s + 0.5 * carry_s   # deploy-relevant: fixed params
    btc = closes["BTC/USDT"].pct_change().loc[lo:hi]

    report("WALK-FORWARD momentum (re-optimised)", equity_of(wf_mom_s))
    report("FIXED-default momentum (same span)", equity_of(fixed_mom_s))
    report("WALK-FORWARD blend (wf-mom + carry)", equity_of(blend_wf))
    report("FIXED-default blend (deploy case)", equity_of(blend_fixed))
    report("BENCH: BTC hold (same span)", equity_of(btc))


if __name__ == "__main__":
    main()
