#!/usr/bin/env python3
"""Run a backtest from a YAML config and print a performance report.

    python run_backtest.py [config.yaml]

Reports in-sample and out-of-sample metrics separately. The out-of-sample (OOS)
block is the one that matters: it's data the strategy parameters were not chosen
on. If OOS looks much worse than in-sample, you've fit noise, not an edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from botcore.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from botcore.data.loader import PERIODS_PER_YEAR, load_ohlcv  # noqa: E402
from botcore.metrics.performance import Metrics, compute_metrics  # noqa: E402
from botcore.risk.sizing import RiskConfig  # noqa: E402
from botcore.strategy import build_strategy  # noqa: E402


def _build_bt_config(bt: dict) -> BacktestConfig:
    risk = bt.get("risk", {})
    return BacktestConfig(
        initial_capital=bt.get("initial_capital", 10_000.0),
        fee_bps=bt.get("fee_bps", 7.5),
        slippage_bps=bt.get("slippage_bps", 2.0),
        risk=RiskConfig(
            risk_per_trade=risk.get("risk_per_trade", 0.01),
            atr_stop_mult=risk.get("atr_stop_mult", 3.0),
            max_leverage=risk.get("max_leverage", 1.0),
        ),
    )


def _print_report(label: str, m: Metrics) -> None:
    print(f"\n=== {label} ===")
    print(f"  Final equity     : {m.final_equity:,.0f}")
    print(f"  Total return     : {m.total_return:+.2%}")
    print(f"  CAGR             : {m.cagr:+.2%}")
    print(f"  Ann. volatility  : {m.ann_volatility:.2%}")
    print(f"  Sharpe           : {m.sharpe:.2f}")
    print(f"  Sortino          : {m.sortino:.2f}")
    print(f"  Max drawdown     : {m.max_drawdown:.2%}")
    print(f"  Calmar           : {m.calmar:.2f}")
    print(f"  Win rate         : {m.win_rate:.1%}")
    print(f"  Profit factor    : {m.profit_factor:.2f}")
    print(f"  Round-trips      : {m.num_trades}")
    print(f"  Exposure         : {m.exposure:.1%}")


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    df = load_ohlcv(cfg)
    timeframe = cfg["data"].get("timeframe", "1h")
    ppy = PERIODS_PER_YEAR.get(timeframe, 8_760)

    strat = build_strategy(cfg["strategy"]["name"], cfg["strategy"].get("params", {}))
    signals = strat.generate(df)
    bt_cfg = _build_bt_config(cfg["backtest"])

    print(f"Data    : {cfg['data']['source']} {cfg['data']['symbol']} {timeframe}  "
          f"({len(df)} bars)")
    print(f"Strategy: {strat}")
    print(f"Costs   : fee {bt_cfg.fee_bps}bps + slippage {bt_cfg.slippage_bps}bps per side")

    # Full-sample run.
    res = run_backtest(df, signals, bt_cfg)
    full = compute_metrics(res.equity, res.returns, res.position, res.trades, ppy)
    _print_report("FULL SAMPLE", full)

    # Out-of-sample split (parameters here are fixed, so this is a clean holdout).
    oos_frac = cfg.get("evaluation", {}).get("oos_fraction", 0.0)
    if 0 < oos_frac < 1:
        split = int(len(df) * (1 - oos_frac))
        for label, sl in [("IN-SAMPLE", slice(0, split)), ("OUT-OF-SAMPLE", slice(split, None))]:
            sub_df = df.iloc[sl]
            sub_sig = signals.iloc[sl]
            r = run_backtest(sub_df, sub_sig, bt_cfg)
            m = compute_metrics(r.equity, r.returns, r.position, r.trades, ppy)
            _print_report(label, m)

    print("\nReminder: synthetic data validates the plumbing, not an edge. "
          "Swap in real data (source: ccxt) before trusting any number above.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
