"""One paper carry decision cycle: snapshot -> target -> (paper) orders -> log.

Same rule as the backtest: hold a delta-neutral long-spot / short-perp pair
whenever trailing funding is positive; size notional to `leverage` x equity,
capped at the leverage the intraday stress test cleared (3x). Flat otherwise.

A "cycle" is meant to be called on a schedule (e.g. every 15 min, and always
just before each 8h funding settlement). It is idempotent w.r.t. funding credit
and persists state, so it is safe to run from cron and to restart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feed import LiveFeed, MarketSnapshot
from .paper_broker import PaperBroker

MAX_LEVERAGE = 3.0  # intraday stress test: 3x safe, 5x liquidates (see Phase 0)


FUNDING_INTERVAL = pd.Timedelta(hours=8)  # Binance USD-M settles every 8h


@dataclass
class CarryConfig:
    leverage: float = 3.0
    # Hysteresis funding band (per-8h fraction). Enter only when trailing funding
    # clears `enter_funding` (well above the near-zero noise floor where a round
    # trip's cost dwarfs the carry); stay in until it drops below `exit_funding`,
    # so a one-settlement wobble across the entry level does not whipsaw us flat.
    # Defaults are the backtest-tuned values (BTC 8h, 2021-23): held 58% of the
    # time, CAGR +32.9% / Sharpe 6.54 @3x vs +23.8% / 4.31 for the old trail>0.
    enter_funding: float = 1e-4   # ~median historical funding (only solid carry)
    exit_funding: float = 2e-5    # noise-floor exit (below this, no edge)
    min_hold: int = 3             # settlements (=1 day) before any exit: let a RT earn out
    basis_kill: float = 0.015     # flatten if perp trades >1.5% above spot (squeeze)
    rebalance_band: float = 0.10  # only resize when target notional drifts >10% of equity


class PaperCarryRunner:
    def __init__(self, feed: LiveFeed, broker: PaperBroker, cfg: CarryConfig | None = None):
        self.feed = feed
        self.broker = broker
        self.cfg = cfg or CarryConfig()
        self.cfg.leverage = min(self.cfg.leverage, MAX_LEVERAGE)

    def _settlements_held(self, snap: MarketSnapshot) -> int:
        """How many 8h settlements the current short has been open (0 if flat)."""
        opened = self.broker.state.perp_opened_ts
        if opened is None:
            return 0
        elapsed = snap.ts - pd.Timestamp(opened)
        return max(0, int(elapsed / FUNDING_INTERVAL))

    def _target_notional(self, snap: MarketSnapshot, equity: float) -> float:
        """Desired delta-neutral notional (USD), with an enter/exit hysteresis
        band and a minimum dwell. 0 => flat."""
        # Hard risk gate first: an active short-squeeze is the one thing that
        # liquidates this trade, so refuse to be short into it (overrides dwell).
        if snap.basis > self.cfg.basis_kill:
            return 0.0

        currently_held = self.broker.state.perp.units < 0
        if currently_held:
            # Stay in until funding drops below the (lower) exit bar, and never
            # exit before the minimum dwell — so each round trip earns its cost.
            if (self._settlements_held(snap) >= self.cfg.min_hold
                    and snap.funding_trail < self.cfg.exit_funding):
                return 0.0
            return self.cfg.leverage * equity
        # Flat: only enter when funding clears the (higher) entry bar.
        if snap.funding_trail >= self.cfg.enter_funding:
            return self.cfg.leverage * equity
        return 0.0

    def cycle(self) -> dict:
        snap = self.feed.snapshot()

        # 1) Credit the funding settlement that just elapsed (if any) while short.
        settled_ts = (snap.next_funding_ts - FUNDING_INTERVAL) if snap.next_funding_ts else snap.ts
        funding_paid = self.broker.accrue_funding(snap.funding_now, snap.perp, settled_ts)

        # 2) Decide target notional.
        equity = self.broker.equity(snap.spot, snap.perp)
        notional = self._target_notional(snap, equity)

        # 3) No-trade band: act on a held<->flat flip, else only when the target
        #    drifts past the band. Stops per-poll churn from equity wobble.
        cur_notional = abs(self.broker.state.spot.units) * snap.spot
        crossing = (notional == 0) != (cur_notional == 0)
        drift = abs(notional - cur_notional) / equity if equity > 0 else 0.0
        if crossing or drift > self.cfg.rebalance_band:
            spot_target = notional / snap.spot          # long spot
            perp_target = -notional / snap.perp         # short perp (delta-neutral)
            fills = [
                self.broker.execute("spot", spot_target, snap.spot, snap.ts),
                self.broker.execute("perp", perp_target, snap.perp, snap.ts),
            ]
            fills = [f for f in fills if f]
        else:
            fills = []

        self.broker.save()
        post_equity = self.broker.equity(snap.spot, snap.perp)

        return {
            "ts": snap.ts.isoformat(),
            "spot": snap.spot, "perp": snap.perp, "basis": snap.basis,
            "funding_now": snap.funding_now, "funding_trail": snap.funding_trail,
            "held": notional > 0, "target_notional": notional,
            "funding_paid": funding_paid, "n_fills": len(fills),
            "equity": post_equity, "funding_accrued": self.broker.state.funding_accrued,
        }
