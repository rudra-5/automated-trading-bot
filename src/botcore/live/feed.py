"""Live market feed for paper trading (read-only ccxt polling).

Phase 1 reads exactly what the carry decision needs and nothing else:
  - current spot price (binance spot)
  - current perp price (binanceusdm)
  - current funding rate + the last few 8h settlements (for the trailing signal)

Everything here is READ-ONLY. No order endpoints are touched. The same data the
backtest consumed from cache is fetched live, so live decisions can be compared
against the backtest one-for-one.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketSnapshot:
    ts: pd.Timestamp          # UTC poll time
    spot: float               # spot mid (last trade)
    perp: float               # perp mark/last
    funding_now: float        # most recent settled 8h funding rate
    funding_trail: float      # mean of last `lookback` settlements (the signal)
    next_funding_ts: pd.Timestamp | None  # when the next funding settles

    @property
    def basis(self) -> float:
        """Perp premium over spot, in fraction (positive => short-squeeze pressure)."""
        return self.perp / self.spot - 1.0


class LiveFeed:
    """Thin read-only wrapper over ccxt spot + USD-M perp clients."""

    def __init__(self, base: str = "BTC", funding_lookback: int = 3):
        import ccxt

        self.base = base
        self.spot_symbol = f"{base}/USDT"
        self.perp_symbol = f"{base}/USDT:USDT"
        self.funding_lookback = funding_lookback
        self._spot = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
        self._perp = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})

    def snapshot(self) -> MarketSnapshot:
        spot_t = self._spot.fetch_ticker(self.spot_symbol)
        perp_t = self._perp.fetch_ticker(self.perp_symbol)

        hist = self._perp.fetch_funding_rate_history(
            self.perp_symbol, limit=self.funding_lookback
        )
        rates = [h["fundingRate"] for h in hist if h.get("fundingRate") is not None]
        funding_now = float(rates[-1]) if rates else 0.0
        funding_trail = float(sum(rates) / len(rates)) if rates else 0.0

        fr = self._perp.fetch_funding_rate(self.perp_symbol)
        next_ts = fr.get("fundingTimestamp") or fr.get("nextFundingTimestamp")
        next_funding_ts = (
            pd.to_datetime(next_ts, unit="ms", utc=True) if next_ts else None
        )

        return MarketSnapshot(
            ts=pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow(),
            spot=float(spot_t["last"]),
            perp=float(perp_t["last"]),
            funding_now=funding_now,
            funding_trail=funding_trail,
            next_funding_ts=next_funding_ts,
        )
