"""Data loading: pluggable source (synthetic | ccxt) with on-disk cache.

The ccxt path is imported lazily so the package has no hard dependency on it —
backtests and CI run on synthetic data, and you only need `pip install ccxt`
when you want real exchange data on a machine with network access.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .synthetic import generate_ohlcv

# Map our timeframe strings to pandas offset aliases for synthetic generation.
_TF_TO_PANDAS = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}

# Approximate number of bars per year for each timeframe (for annualisation).
PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


def _cache_path(cache_dir: str, source: str, exchange: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    name = f"{source}_{exchange}_{safe_symbol}_{timeframe}.parquet"
    return Path(cache_dir) / name


def load_ohlcv(cfg: dict, use_cache: bool = True) -> pd.DataFrame:
    """Load OHLCV per the `data` config block. Returns a UTC-indexed DataFrame
    with columns: open, high, low, close, volume."""
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    exchange = data_cfg.get("exchange", "binance")
    symbol = data_cfg.get("symbol", "BTC/USDT")
    timeframe = data_cfg.get("timeframe", "1h")
    cache_dir = data_cfg.get("cache_dir", "data/cache")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(cache_dir, source, exchange, symbol, timeframe)

    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    if source == "synthetic":
        s = data_cfg.get("synthetic", {})
        df = generate_ohlcv(
            bars=s.get("bars", 8760),
            start_price=s.get("start_price", 30_000.0),
            annual_drift=s.get("annual_drift", 0.20),
            annual_vol=s.get("annual_vol", 0.70),
            periods_per_year=PERIODS_PER_YEAR.get(timeframe, 8_760),
            seed=s.get("seed", 42),
            start=data_cfg.get("start", "2021-01-01"),
            timeframe_pandas=_TF_TO_PANDAS.get(timeframe, "1h"),
        )
    elif source == "ccxt":
        df = _fetch_ccxt(
            exchange, symbol, timeframe,
            start=data_cfg.get("start"), end=data_cfg.get("end"),
        )
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    df.to_parquet(cache_file)
    return df


def _fetch_ccxt(exchange: str, symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Fetch real OHLCV via ccxt, paging through the exchange's limit.

    Run this on a machine with network + exchange access. Many exchanges
    geo-block cloud IPs (you'll see HTTP 403); use one available in your region.
    """
    try:
        import ccxt  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "ccxt is required for source: ccxt. Install with `pip install ccxt`."
        ) from e

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since = ex.parse8601(f"{start}T00:00:00Z") if start else None
    end_ms = ex.parse8601(f"{end}T00:00:00Z") if end else None
    tf_ms = ex.parse_timeframe(timeframe) * 1000

    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
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
