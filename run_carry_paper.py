#!/usr/bin/env python3
"""Phase 1: live PAPER trading of the BTC carry sleeve. Places NO real orders.

Polls live Binance spot + USD-M perp, runs the same carry decision the backtest
used (delta-neutral, hold when trailing funding > 0, 3x cap from the intraday
stress test), and records intended fills to a paper blotter. Equity is marked to
the live market each cycle so live behaviour can be reconciled against backtest.

Usage:
    python run_carry_paper.py            # loop every 15 min (default), Ctrl-C to stop
    python run_carry_paper.py --loop 0   # run a single cycle and exit
    python run_carry_paper.py --loop 300 # custom interval (seconds)

Nothing here touches an order endpoint. This is the watch-only stage that must
run clean for weeks before any real money (Phase 3) is considered.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from botcore.live.carry_runner import CarryConfig, PaperCarryRunner
from botcore.live.feed import LiveFeed
from botcore.live.paper_broker import PaperBroker


def _print_cycle(d: dict) -> None:
    flag = "HELD " if d["held"] else "FLAT "
    print(
        f"{d['ts'][:19]}  {flag} "
        f"spot={d['spot']:,.1f} perp={d['perp']:,.1f} basis={d['basis']:+.3%}  "
        f"f_now={d['funding_now']:+.4%} f_trail={d['funding_trail']:+.4%}  "
        f"fills={d['n_fills']} fund_paid={d['funding_paid']:+,.2f}  "
        f"equity={d['equity']:,.2f} (funding_total={d['funding_accrued']:+,.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=900,
                    help="seconds between cycles; default 900 (15 min), pass 0 to run a single cycle")
    ap.add_argument("--leverage", type=float, default=3.0)
    args = ap.parse_args()

    feed = LiveFeed(base="BTC")
    broker = PaperBroker()
    runner = PaperCarryRunner(feed, broker, CarryConfig(leverage=args.leverage))

    print("PAPER MODE — no real orders. State: state/paper_carry.json  Blotter: logs/paper_blotter.csv")
    if args.loop <= 0:
        try:
            _print_cycle(runner.cycle())
        except Exception as e:  # surface the cause in the log, then fail loudly
            print(f"CYCLE FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            sys.exit(1)
        return

    print(f"Looping every {args.loop}s. Ctrl-C to stop.")
    consecutive_errors = 0
    try:
        while True:
            try:
                _print_cycle(runner.cycle())
                consecutive_errors = 0
            except Exception as e:  # transient network/exchange blip: skip, never die
                consecutive_errors += 1
                print(f"[WARN] cycle failed ({consecutive_errors}): "
                      f"{type(e).__name__}: {str(e)[:120]} — skipping, state preserved",
                      flush=True)
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print("\nStopped. State saved.")


if __name__ == "__main__":
    main()
