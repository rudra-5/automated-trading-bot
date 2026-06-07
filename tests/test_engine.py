import numpy as np
import pandas as pd

from botcore.backtest.engine import BacktestConfig, run_backtest
from botcore.risk.sizing import RiskConfig


def _make_df(prices):
    idx = pd.date_range("2021-01-01", periods=len(prices), freq="1h", tz="UTC")
    p = np.array(prices, dtype=float)
    return pd.DataFrame({
        "open": p, "high": p * 1.001, "low": p * 0.999, "close": p,
    }, index=idx)


def test_no_signal_keeps_capital_constant():
    df = _make_df([100, 101, 102, 103, 104])
    sig = pd.DataFrame({"signal": [0, 0, 0, 0, 0], "atr": [1.0] * 5}, index=df.index)
    cfg = BacktestConfig(initial_capital=10_000, fee_bps=10, slippage_bps=5)
    res = run_backtest(df, sig, cfg)
    assert np.allclose(res.equity.values, 10_000)


def test_fees_reduce_equity_on_round_trip():
    # Flat price, one entry then exit: with costs we must end below start.
    df = _make_df([100, 100, 100, 100, 100, 100])
    sig = pd.DataFrame({"signal": [0, 1, 1, 0, 0, 0], "atr": [2.0] * 6}, index=df.index)
    with_fees = run_backtest(df, sig, BacktestConfig(fee_bps=10, slippage_bps=5))
    no_fees = run_backtest(df, sig, BacktestConfig(fee_bps=0, slippage_bps=0))
    assert with_fees.equity.iloc[-1] < no_fees.equity.iloc[-1]
    # With zero costs and flat price, equity returns to ~initial.
    assert abs(no_fees.equity.iloc[-1] - no_fees.config.initial_capital) < 1e-6


def test_long_profits_when_price_rises():
    df = _make_df([100, 100, 110, 120, 130, 130])
    sig = pd.DataFrame({"signal": [0, 1, 1, 1, 0, 0], "atr": [2.0] * 6}, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(fee_bps=0, slippage_bps=0))
    assert res.equity.iloc[-1] > res.config.initial_capital


def test_stop_is_respected():
    # Enter long at open=100 with ATR=2, stop_mult=3 -> stop at 94.
    # Then a bar dives to low=90 should trigger the stop, capping the loss.
    df = _make_df([100, 100, 100, 80, 80])
    df.iloc[3, df.columns.get_loc("low")] = 90.0  # ensure low pierces the stop
    sig = pd.DataFrame({"signal": [0, 1, 1, 1, 1], "atr": [2.0] * 5}, index=df.index)
    cfg = BacktestConfig(fee_bps=0, slippage_bps=0,
                         risk=RiskConfig(risk_per_trade=0.01, atr_stop_mult=3.0))
    res = run_backtest(df, sig, cfg)
    # A stop fill must be recorded.
    assert (res.trades["reason"] == "stop_long").any()


def test_no_lookahead_last_bar_signal_ignored():
    # A signal only on the final bar can never be executed (no next bar),
    # so equity must be unchanged from a never-trading run.
    df = _make_df([100, 100, 100, 100])
    sig = pd.DataFrame({"signal": [0, 0, 0, 1], "atr": [2.0] * 4}, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(fee_bps=10, slippage_bps=5))
    assert np.allclose(res.equity.values, res.config.initial_capital)
