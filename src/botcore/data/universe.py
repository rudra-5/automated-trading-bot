"""Multi-asset universe loader: fetch + cache daily closes for many symbols.

Cross-sectional strategies need a *panel* of prices (many coins, aligned on the
same dates) rather than one symbol's OHLCV. This module fetches each symbol via
the existing ccxt path, caches it to parquet, and assembles a single DataFrame
of aligned closes (index = dates, one column per symbol).

Symbols that don't have history back to the requested start are dropped, so the
panel is survivorship-aware only in the sense of "listed early enough" — it does
NOT correct for coins that later delisted. Keep that caveat in mind.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _fetch_one_daily(exchange: str, symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch a single symbol's daily OHLCV via ccxt, paging to the start date."""
    import ccxt  # lazy import; only needed for real data

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since = ex.parse8601(f"{start}T00:00:00Z")
    end_ms = ex.parse8601(f"{end}T00:00:00Z") if end else None
    tf_ms = ex.parse_timeframe("1d") * 1000

    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if end_ms and since >= end_ms:
            break
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    if end_ms:
        df = df[df.index < pd.to_datetime(end_ms, unit="ms", utc=True)]
    return df


def load_universe(
    symbols: list[str],
    exchange: str = "binance",
    start: str = "2021-01-01",
    end: str | None = "2024-01-01",
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of daily closes: index = UTC dates, columns = symbols.

    Each symbol is cached individually so re-runs and partial universes are cheap.
    Symbols lacking history at `start` (first bar > start + 5d) are dropped with a
    printed note.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_ts = pd.to_datetime(start, utc=True)
    closes: dict[str, pd.Series] = {}

    for sym in symbols:
        safe = sym.replace("/", "-")
        cache_file = Path(cache_dir) / f"universe_{exchange}_{safe}_1d.parquet"
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            df = _fetch_one_daily(exchange, sym, start, end)
            df.to_parquet(cache_file)

        if df.empty:
            print(f"  drop {sym}: no data")
            continue
        first = df.index[0]
        if first > start_ts + pd.Timedelta(days=5):
            print(f"  drop {sym}: history only from {first.date()} (after {start})")
            continue
        closes[sym] = df["close"]

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.loc[start_ts:]
    return panel
