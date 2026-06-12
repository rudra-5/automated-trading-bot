#!/usr/bin/env python3
"""A/B structural momentum variants on the honest yardstick.

The walk-forward lesson: improvements must show up out-of-sample across regimes,
not in a tuned backtest. So we judge every variant by the fixed-params blend
(momentum + honest carry) over the cross-regime span 2022-01-01 .. 2023-12-21
(the walk-forward OOS window, which includes the 2022 bear). No re-optimisation:
each variant is a sensible structural default, measured the same way.
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
OOS_LO, OOS_HI = "2022-01-01", "2023-12-21"

VARIANTS = {
    "baseline (equal, 30d)":        dict(),
    "inv-vol weight":               dict(weighting="invvol"),
    "multi-horizon (20/40/60)":     dict(lookbacks=(20, 40, 60)),
    "inv-vol + multi-horizon":      dict(weighting="invvol", lookbacks=(20, 40, 60)),
}


def vol_target(rets, target, lookback=30, max_lev=4.0, ppy=PPY):
    daily_target = target / np.sqrt(ppy)
    roll = rets.rolling(lookback).std().shift(1)
    lev = (daily_target / roll).replace([np.inf, -np.inf], np.nan).clip(upper=max_lev).fillna(0.0)
    return lev * rets


def stats(rets, ppy=PPY):
    r = rets.fillna(0.0)
    eq = (1.0 + r).cumprod()
    years = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=0) * np.sqrt(ppy)
    sh = r.mean() * ppy / vol if vol > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    return cagr, vol, sh, dd


def main() -> None:
    closes = load_universe(UNIVERSE)
    funding = load_funding(UNIVERSE).reindex(closes.index).fillna(0.0)
    perp = load_perp_closes(UNIVERSE).reindex(closes.index)
    cfg = PortfolioConfig(cost_bps=9.5, target_vol=SLEEVE_VOL, max_leverage=3.0, periods_per_year=PPY)

    carry = vol_target(
        carry_returns_honest(funding, closes, perp, 7, 7, cost_bps=6.0, margin_fraction=0.30),
        SLEEVE_VOL, max_lev=2.0,
    )

    print(f"Honest yardstick: fixed-params 50/50 blend, span {OOS_LO}..{OOS_HI}\n")
    print(f"{'Variant':<28}{'Sharpe':>8}{'CAGR':>9}{'Vol':>8}{'MaxDD':>9}")
    print("-" * 62)
    for name, kw in VARIANTS.items():
        w = momentum_weights(closes, lookback=30, skip=2, top_k=5, rebalance_days=7,
                            regime_ma=50, **kw)
        mom = run_portfolio(closes, w, cfg).returns
        common = mom.index.intersection(carry.index)
        blend = 0.5 * mom.reindex(common).fillna(0.0) + 0.5 * carry.reindex(common).fillna(0.0)
        cagr, vol, sh, dd = stats(blend.loc[OOS_LO:OOS_HI])
        print(f"{name:<28}{sh:>8.2f}{cagr:>+8.1%}{vol:>8.1%}{dd:>+8.1%}")


if __name__ == "__main__":
    main()
