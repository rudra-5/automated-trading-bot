"""Funding-rate loader for perpetual futures (the carry sleeve's raw input).

Binance USD-M perps settle funding every 8 hours (3x/day). When funding is
positive, longs pay shorts; that payment, collected by a short-perp leg that is
delta-hedged with long spot, is the carry return.

This module fetches each perp's funding-rate history, caches it, and assembles a
DAILY funding panel: index = date, columns = base symbol, value = the day's
total funding rate (sum of the three 8h settlements). A long-spot/short-perp
position in that coin earns ~that daily rate on its notional, with the price
move hedged away.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _fetch_one_funding(perp: str, start: str, end: str | None) -> pd.DataFrame:
    """Page through one perp's 8h funding history via ccxt (binanceusdm)."""
    import ccxt

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    since = ex.parse8601(f"{start}T00:00:00Z")
    end_ms = ex.parse8601(f"{end}T00:00:00Z") if end else None

    rows: list[dict] = []
    while True:
        batch = ex.fetch_funding_rate_history(perp, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        since = last_ts + 1
        if end_ms and since >= end_ms:
            break
        if len(batch) < 1000:
            break

    df = pd.DataFrame(
        [{"ts": r["timestamp"], "rate": r["fundingRate"]} for r in rows]
    )
    if df.empty:
        return df
    df = df.drop_duplicates("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    if end_ms:
        df = df[df.index < pd.to_datetime(end_ms, unit="ms", utc=True)]
    return df


def load_funding(
    symbols: list[str],
    start: str = "2021-01-01",
    end: str | None = "2024-01-01",
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DAILY funding-rate panel: index=date, cols=symbols (e.g. BTC/USDT).

    `symbols` are spot-style names (BTC/USDT); the matching perp is BTC/USDT:USDT.
    Each day's value is the sum of that day's three 8h funding settlements.
    """
    os.makedirs(cache_dir, exist_ok=True)
    daily: dict[str, pd.Series] = {}

    for sym in symbols:
        base = sym.split("/")[0]
        perp = f"{base}/USDT:USDT"
        cache_file = Path(cache_dir) / f"funding_binance_{base}_8h.parquet"
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = _fetch_one_funding(perp, start, end)
            if not df.empty:
                df.to_parquet(cache_file)
        if df.empty:
            print(f"  drop funding {sym}: none")
            continue
        # Sum the three 8h rates within each UTC day -> daily funding yield.
        daily[sym] = df["rate"].resample("1D").sum()

    panel = pd.DataFrame(daily).sort_index()
    panel = panel.loc[pd.to_datetime(start, utc=True):]
    return panel


def _fetch_perp_daily(perp: str, start: str, end: str | None) -> pd.DataFrame:
    """Page through one perp's daily OHLCV via ccxt (binanceusdm)."""
    import ccxt

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    since = ex.parse8601(f"{start}T00:00:00Z")
    end_ms = ex.parse8601(f"{end}T00:00:00Z") if end else None
    tf_ms = ex.parse_timeframe("1d") * 1000

    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(perp, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if (end_ms and since >= end_ms) or len(batch) < 1000:
            break

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    if end_ms:
        df = df[df.index < pd.to_datetime(end_ms, unit="ms", utc=True)]
    return df


def load_perp_closes(
    symbols: list[str],
    start: str = "2021-01-01",
    end: str | None = "2024-01-01",
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily perp close panel (index=date, cols=spot-style symbol e.g. BTC/USDT).

    Needed to model the carry sleeve's basis PnL: the long-spot/short-perp hedge
    is imperfect, and (spot_ret - perp_ret) is the realised tracking error that
    blows out in stress. Without it, carry looks risk-free (it isn't).
    """
    os.makedirs(cache_dir, exist_ok=True)
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        base = sym.split("/")[0]
        perp = f"{base}/USDT:USDT"
        cache_file = Path(cache_dir) / f"perp_binance_{base}_1d.parquet"
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = _fetch_perp_daily(perp, start, end)
            if not df.empty:
                df.to_parquet(cache_file)
        if not df.empty:
            closes[sym] = df["close"]
    panel = pd.DataFrame(closes).sort_index()
    return panel.loc[pd.to_datetime(start, utc=True):]
