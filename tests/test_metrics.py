import numpy as np
import pandas as pd

from botcore.metrics.performance import compute_metrics, _max_drawdown


def _idx(n):
    return pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")


def test_max_drawdown_simple():
    eq = pd.Series([100, 120, 60, 90], index=_idx(4))
    # Peak 120 -> trough 60 = -50%
    assert _max_drawdown(eq) == -0.5


def test_total_return_and_final_equity():
    eq = pd.Series([100, 110, 121], index=_idx(3))
    rets = eq.pct_change().fillna(0)
    pos = pd.Series([1, 1, 1], index=_idx(3))
    m = compute_metrics(eq, rets, pos, pd.DataFrame(), periods_per_year=8760)
    assert abs(m.total_return - 0.21) < 1e-9
    assert m.final_equity == 121
    assert abs(m.exposure - 1.0) < 1e-9


def test_flat_curve_has_zero_vol_and_sharpe():
    eq = pd.Series([100.0] * 50, index=_idx(50))
    rets = eq.pct_change().fillna(0)
    pos = pd.Series([0] * 50, index=_idx(50))
    m = compute_metrics(eq, rets, pos, pd.DataFrame(), periods_per_year=8760)
    assert m.ann_volatility == 0.0
    assert m.sharpe == 0.0
    assert m.exposure == 0.0


def test_profit_factor_from_blotter():
    # One winning round trip (+10) and one losing (-5): PF = 10/5 = 2.0
    idx = _idx(4)
    trades = pd.DataFrame({
        "delta_units": [1.0, -1.0, 1.0, -1.0],
        "price": [100.0, 110.0, 100.0, 95.0],
        "fee": [0.0, 0.0, 0.0, 0.0],
    }, index=idx)
    eq = pd.Series([100, 100, 100, 100], index=idx)
    rets = eq.pct_change().fillna(0)
    pos = pd.Series([1, 0, 1, 0], index=idx)
    m = compute_metrics(eq, rets, pos, trades, periods_per_year=8760)
    assert m.num_trades == 2
    assert m.win_rate == 0.5
    assert abs(m.profit_factor - 2.0) < 1e-9
