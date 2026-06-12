#!/usr/bin/env python3
"""BTC-only carry stress test on intraday (1h) data.

The daily carry sleeve caps leverage at 2x because daily close-to-close bars
can't see intraday liquidation cascades: the perp can spike above spot mid-day,
wipe the short-perp leg's isolated margin, and mean-revert by the daily close —
the daily bar shows ~0, but the position was already liquidated.

Here we replay BTC carry at the native 8h funding cadence and use 1h HIGHS to
measure the true intra-window max-adverse-upside excursion that stresses the
short-perp leg. We sweep sleeve leverage L and find where intraday liquidations
start, i.e. whether the 2x cap is too conservative.

Phase 0 upgrade: real BTC *perp* 1h is now cached, so the short-perp leg's
adverse move and the spot-perp basis are measured directly (no spot proxy). We
also print the proxy gap so the earlier spot-based run can be judged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

PPY_8H = 365 * 3  # 8h settlements per year
MAINT_MARGIN = 0.005  # exchange maintenance margin (~0.5% for BTC perp)
COST_BPS = 6.0
LEGS = 2


def _to_8h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLC into 8h windows anchored at 00:00/08:00/16:00 UTC."""
    return df_1h.resample("8h", origin="epoch").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna()


def load_btc_intraday() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (8h spot, 8h perp, 1h basis frame, 8h funding)."""
    spot_1h = pd.read_parquet("data/cache/ccxt_binance_BTC-USDT_1h.parquet")
    perp_1h = pd.read_parquet("data/cache/ccxt_binanceusdm_BTC-USDT-USDT_1h.parquet")
    funding_8h = pd.read_parquet("data/cache/funding_binance_BTC_8h.parquet")["rate"]
    funding_8h.index = funding_8h.index.floor("h")  # settlements land on 00/08/16:00

    # Hourly basis = perp/spot - 1; intra-window max gauges short squeezes.
    common = spot_1h.index.intersection(perp_1h.index)
    basis_1h = pd.DataFrame({
        "basis": perp_1h.loc[common, "close"] / spot_1h.loc[common, "close"] - 1.0,
    })
    return _to_8h(spot_1h), _to_8h(perp_1h), basis_1h, funding_8h


def build_carry_8h(spot: pd.DataFrame, perp: pd.DataFrame, basis_1h: pd.DataFrame,
                   funding: pd.Series, select_lookback: int = 3) -> pd.DataFrame:
    """Per-8h carry book: held flag, net funding, basis PnL, real short-leg excursion.

    Position = long-spot / short-perp whenever trailing-mean funding (causal,
    lagged) is positive. Earn that settlement's funding on the held position.
    `up_excursion` uses the *perp* window high (real short-leg adverse move).
    `basis_pnl` = spot_ret - perp_ret close-to-close (the imperfect-hedge term).
    """
    idx = spot.index.intersection(perp.index)
    f = funding.reindex(idx).fillna(0.0)
    trail = f.rolling(select_lookback).mean().shift(1)
    held = (trail > 0).astype(float)  # causal: yesterday's signal

    turnover = held.diff().abs().fillna(held)
    cost = turnover * (COST_BPS / 1e4) * LEGS

    gross_funding = held * f
    spot_ret = spot["close"].reindex(idx).pct_change()
    perp_ret = perp["close"].reindex(idx).pct_change()
    basis_pnl = held * (spot_ret - perp_ret).fillna(0.0)  # delta-neutral tracking error

    up_excursion = (perp["high"].reindex(idx) / perp["open"].reindex(idx) - 1.0).clip(lower=0.0)
    spot_excursion = (spot["high"].reindex(idx) / spot["open"].reindex(idx) - 1.0).clip(lower=0.0)
    intraday_basis_max = basis_1h["basis"].resample("8h", origin="epoch").max().reindex(idx)

    return pd.DataFrame({
        "held": held, "funding_net": gross_funding - cost, "basis_pnl": basis_pnl,
        "up_excursion": up_excursion, "spot_excursion": spot_excursion,
        "intraday_basis_max": intraday_basis_max,
    })


def simulate(book: pd.DataFrame, leverage: float) -> dict:
    """Compound the levered carry book with an intraday liquidation overlay.

    Short-perp leg posts equity as margin at notional = L*equity, so it liquidates
    when the intra-window upside move >= 1/L - maintenance. On liquidation the
    leg's margin (≈100% of deployed equity) is lost: model as a -100% period.
    """
    liq_threshold = 1.0 / leverage - MAINT_MARGIN
    held = book["held"].values
    pnl = (book["funding_net"] + book["basis_pnl"]).values  # funding + basis tracking error
    upx = book["up_excursion"].values

    liquidated = (held > 0) & (upx >= liq_threshold)
    period_ret = np.where(liquidated, -1.0, leverage * pnl)

    rets = pd.Series(period_ret, index=book.index)
    equity = 10_000.0 * (1.0 + rets).cumprod()
    equity = equity.clip(lower=0.0)

    n = len(rets)
    years = n / PPY_8H
    final = float(equity.iloc[-1])
    cagr = (final / 10_000.0) ** (1 / years) - 1 if final > 0 else -1.0
    ann_vol = float(rets.std(ddof=0) * np.sqrt(PPY_8H))
    sharpe = (float(rets.mean()) * PPY_8H) / ann_vol if ann_vol > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())

    return {
        "L": leverage, "liq_threshold": liq_threshold,
        "n_liquidations": int(liquidated.sum()),
        "cagr": cagr, "sharpe": sharpe, "ann_vol": ann_vol,
        "max_dd": dd, "final": final,
    }


def main() -> None:
    spot, perp, basis_1h, funding = load_btc_intraday()
    book = build_carry_8h(spot, perp, basis_1h, funding)
    print(f"BTC 8h carry book: {len(book)} settlements "
          f"({book.index[0].date()} -> {book.index[-1].date()}), "
          f"held {book['held'].mean():.0%} of the time")
    print(f"PERP upside excursion (per 8h window): "
          f"median {book['up_excursion'].median():.2%}, "
          f"p99 {book['up_excursion'].quantile(0.99):.2%}, "
          f"max {book['up_excursion'].max():.2%}")
    print(f"  proxy gap: SPOT excursion max {book['spot_excursion'].max():.2%} "
          f"vs PERP max {book['up_excursion'].max():.2%} "
          f"(perp {'wider' if book['up_excursion'].max() > book['spot_excursion'].max() else 'tighter'})")
    print(f"Intraday basis (perp/spot-1): max {book['intraday_basis_max'].max():+.2%}, "
          f"p99 {book['intraday_basis_max'].quantile(0.99):+.2%}  "
          f"[positive = perp above spot = short-squeeze pressure]")
    print(f"Close-to-close basis PnL per 8h: mean {book['basis_pnl'].mean():+.4%}, "
          f"worst {book['basis_pnl'].min():+.2%}")

    print("\n  L   liq@move   #liq   CAGR      Sharpe   AnnVol   MaxDD     FinalEq")
    print("  " + "-" * 70)
    for L in [2, 3, 5, 8, 10, 15, 20, 30]:
        m = simulate(book, L)
        print(f"  {m['L']:>2.0f}  {m['liq_threshold']:>7.1%}  {m['n_liquidations']:>5d}  "
              f"{m['cagr']:>+7.2%}  {m['sharpe']:>6.2f}  {m['ann_vol']:>6.2%}  "
              f"{m['max_dd']:>+7.2%}  {m['final']:>10,.0f}")

    # Highest L with zero intraday liquidations = the empirically safe cap.
    safe = [L for L in range(2, 31) if simulate(book, L)["n_liquidations"] == 0]
    if safe:
        best = max(safe)
        m = simulate(book, best)
        print(f"\nMax leverage with ZERO intraday liquidations: {best}x "
              f"-> CAGR {m['cagr']:+.2%}, Sharpe {m['sharpe']:.2f}, MaxDD {m['max_dd']:+.2%}")


if __name__ == "__main__":
    main()
