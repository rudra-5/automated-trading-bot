"""Paper broker: logs INTENDED orders, places nothing real.

This is the safety boundary for Phase 1. Every order is "filled" against the
current market price plus a modelled slippage, charged the same per-side fee the
backtest used, and recorded to a blotter. No ccxt order endpoint is ever called.

State (positions, cash, blotter) persists to a JSON file so a multi-week paper
run survives restarts. Swapping this class for a real broker later changes one
seam and nothing else.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Leg:
    """One side of the delta-neutral pair (spot long or perp short)."""
    units: float = 0.0        # signed: +long, -short
    avg_price: float = 0.0    # entry VWAP


@dataclass
class PaperState:
    cash: float = 10_000.0
    funding_accrued: float = 0.0
    spot: Leg = field(default_factory=Leg)
    perp: Leg = field(default_factory=Leg)
    last_funding_ts: str | None = None   # ISO of last settlement credited (idempotency)
    perp_opened_ts: str | None = None    # ISO when the current short was opened


class PaperBroker:
    def __init__(self, state_path: str = "state/paper_carry.json",
                 blotter_path: str = "logs/paper_blotter.csv",
                 cost_bps: float = 6.0, slippage_bps: float = 2.0,
                 initial_cash: float = 10_000.0):
        self.state_path = Path(state_path)
        self.blotter_path = Path(blotter_path)
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.state = self._load(initial_cash)

    # --- persistence ---
    def _load(self, initial_cash: float) -> PaperState:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            return PaperState(
                cash=raw["cash"], funding_accrued=raw["funding_accrued"],
                spot=Leg(**raw["spot"]), perp=Leg(**raw["perp"]),
                last_funding_ts=raw.get("last_funding_ts"),
                perp_opened_ts=raw.get("perp_opened_ts"),
            )
        return PaperState(cash=initial_cash)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "cash": self.state.cash, "funding_accrued": self.state.funding_accrued,
            "spot": asdict(self.state.spot), "perp": asdict(self.state.perp),
            "last_funding_ts": self.state.last_funding_ts,
            "perp_opened_ts": self.state.perp_opened_ts,
        }, indent=2))

    # --- the only mutation: a (paper) fill ---
    def execute(self, leg_name: str, target_units: float, market_price: float,
                ts: pd.Timestamp) -> dict | None:
        """Move `leg_name` to target_units at market + slippage. Logs, never sends.

        Returns the fill dict (or None if no change). Slippage always hurts: buys
        fill above, sells below.
        """
        leg = getattr(self.state, leg_name)
        delta = target_units - leg.units
        if abs(delta) < 1e-9:
            return None

        side = 1 if delta > 0 else -1
        fill_price = market_price * (1 + side * self.slippage_bps / 1e4)
        notional = abs(delta) * fill_price
        fee = notional * self.cost_bps / 1e4

        # Cash treatment differs by instrument:
        #  - spot is bought/sold outright: notional moves cash.
        #  - perp is margin-based: opening notional does NOT move cash; only
        #    realised PnL (when the position is reduced) and fees do.
        if leg_name == "spot":
            self.state.cash -= delta * fill_price + fee
        else:
            reducing = leg.units != 0 and (delta > 0) != (leg.units > 0)
            if reducing:
                covered = min(abs(delta), abs(leg.units))
                # Short profit when fill is below entry: (avg - fill) * qty.
                realised = (leg.avg_price - fill_price) * covered * (1 if leg.units < 0 else -1)
                self.state.cash += realised
            self.state.cash -= fee

        # Update leg VWAP (only when adding in the same direction).
        if leg.units == 0 or (leg.units > 0) == (delta > 0):
            total = leg.units + delta
            leg.avg_price = (
                (leg.avg_price * leg.units + fill_price * delta) / total
                if abs(total) > 1e-12 else 0.0
            )
        was_flat = leg.units == 0
        leg.units = target_units

        # Stamp / clear the short-open time so funding accrues only over the
        # settlements that fully elapse while the short is on.
        if leg_name == "perp":
            if was_flat and leg.units < 0:
                self.state.perp_opened_ts = ts.isoformat()
            elif abs(leg.units) < 1e-12:
                self.state.perp_opened_ts = None

        fill = {
            "ts": ts.isoformat(), "leg": leg_name, "side": "BUY" if side > 0 else "SELL",
            "delta_units": delta, "fill_price": fill_price, "fee": fee,
            "PAPER": True,
        }
        self._append_blotter(fill)
        return fill

    def accrue_funding(self, funding_rate: float, perp_price: float,
                       settled_ts: pd.Timestamp) -> float:
        """Credit the funding settlement that just elapsed, to the short-perp leg.

        `settled_ts` is the timestamp of the most recent *past* settlement. We
        credit only if the short was already open before it and we have not
        credited it yet, so each real settlement pays exactly once and never on
        the bar we entered. A short perp receives funding when the rate is >0.
        """
        if self.state.perp.units >= 0:
            return 0.0
        ts_iso = settled_ts.isoformat()
        if self.state.last_funding_ts == ts_iso:
            return 0.0
        opened = self.state.perp_opened_ts
        if opened is None or pd.Timestamp(opened) >= settled_ts:
            self.state.last_funding_ts = ts_iso  # mark seen; we weren't short for it
            return 0.0
        short_notional = abs(self.state.perp.units) * perp_price
        payment = short_notional * funding_rate
        self.state.cash += payment
        self.state.funding_accrued += payment
        self.state.last_funding_ts = ts_iso
        return payment

    def equity(self, spot_price: float, perp_price: float) -> float:
        """Mark-to-market: cash + spot leg value + perp PnL.

        perp.units is signed (short => negative), so signed PnL is simply
        units * (price - entry): a short loses as price rises. This must move
        OPPOSITE to the spot leg, leaving equity flat to BTC direction.
        """
        spot_val = self.state.spot.units * spot_price
        perp_pnl = self.state.perp.units * (perp_price - self.state.perp.avg_price)
        return self.state.cash + spot_val + perp_pnl

    def _append_blotter(self, fill: dict) -> None:
        self.blotter_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.blotter_path.exists()
        pd.DataFrame([fill]).to_csv(self.blotter_path, mode="a", header=header, index=False)
