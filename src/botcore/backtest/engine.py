"""Bar-by-bar backtest engine with realistic frictions.

Design choices that keep results honest:
  * No lookahead: a bar's signal (decided on its close) is executed at the NEXT
    bar's open. We implement this by acting on `signal[t-1]` at `open[t]`.
  * Costs on every fill: taker fee (bps) + slippage (bps) applied to traded
    notional. This is what kills most "profitable" backtests on contact.
  * ATR stops checked intrabar against this bar's low/high. If price gapped
    through the stop at the open, we fill at the open (worse), not the stop.
  * No leverage by default; sizing from the risk module is equity-aware.

The engine is long/short capable and marks equity to the bar close. It returns
an equity curve, a trade blotter, and the per-bar position for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..risk.sizing import RiskConfig, size_position


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 7.5         # per-side taker fee, basis points (0.075%)
    slippage_bps: float = 2.0    # per-side slippage assumption, basis points
    risk: RiskConfig = field(default_factory=RiskConfig)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """Run the backtest.

    df:      OHLCV (open, high, low, close) UTC-indexed.
    signals: DataFrame with 'signal' (target direction -1/0/+1) and 'atr'.
    """
    data = df.join(signals[["signal", "atr"]], how="inner")
    open_ = data["open"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    signal = data["signal"].fillna(0).to_numpy(dtype=float)
    atr_arr = data["atr"].to_numpy(dtype=float)
    index = data.index
    n = len(data)

    fee = cfg.fee_bps / 1e4
    slip = cfg.slippage_bps / 1e4
    risk = cfg.risk

    cash = cfg.initial_capital
    units = 0.0
    entry_price = 0.0
    stop_price = np.nan
    equity_curve = np.empty(n)
    trades: list[dict] = []

    def fill(target_units: float, ref_price: float, ts, reason: str) -> None:
        """Trade from current `units` to `target_units` at ref_price, charging
        slippage (price moves against us) and fee on the traded notional."""
        nonlocal cash, units, entry_price
        delta = target_units - units
        if delta == 0:
            return
        # Slippage worsens the execution price in the direction of the trade.
        exec_price = ref_price * (1 + slip) if delta > 0 else ref_price * (1 - slip)
        notional = abs(delta) * exec_price
        cash -= delta * exec_price       # buying reduces cash, selling adds
        cash -= notional * fee           # fee always a cost
        trades.append({
            "ts": ts, "reason": reason, "delta_units": delta,
            "price": exec_price, "fee": notional * fee, "units_after": target_units,
        })
        units = target_units
        if target_units != 0:
            entry_price = exec_price

    for t in range(n):
        px_open = open_[t]

        # 1) Stop check against this bar's range (before acting on new signal).
        if units != 0 and not np.isnan(stop_price):
            if units > 0 and low[t] <= stop_price:
                ref = min(px_open, stop_price)  # gap-through fills at open
                fill(0.0, ref, index[t], "stop_long")
                stop_price = np.nan
            elif units < 0 and high[t] >= stop_price:
                ref = max(px_open, stop_price)
                fill(0.0, ref, index[t], "stop_short")
                stop_price = np.nan

        # 2) Act on the PREVIOUS bar's signal at this bar's open (no lookahead).
        desired_dir = int(np.sign(signal[t - 1])) if t > 0 else 0
        current_dir = int(np.sign(units))

        if desired_dir != current_dir:
            if desired_dir == 0:
                if units != 0:
                    fill(0.0, px_open, index[t], "exit_signal")
                    stop_price = np.nan
            else:
                # Close any opposite position, then open the new one.
                if units != 0:
                    fill(0.0, px_open, index[t], "reverse_close")
                equity_now = cash  # flat here, so equity == cash
                new_units, new_stop = size_position(
                    equity_now, px_open, atr_arr[t], desired_dir, risk,
                )
                if new_units != 0:
                    fill(new_units, px_open, index[t], "entry")
                    stop_price = new_stop

        # 3) Mark to market on the close.
        equity_curve[t] = cash + units * close[t]

    equity = pd.Series(equity_curve, index=index, name="equity")
    returns = equity.pct_change().fillna(0.0)
    position = pd.Series(np.sign(signal), index=index, name="position").shift(1).fillna(0)
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.set_index("ts")

    return BacktestResult(
        equity=equity, returns=returns, position=position,
        trades=trades_df, config=cfg,
    )
