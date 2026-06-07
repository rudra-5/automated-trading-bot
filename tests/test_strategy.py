import numpy as np

from botcore.data.synthetic import generate_ohlcv
from botcore.strategy import build_strategy


def test_momentum_output_contract():
    df = generate_ohlcv(bars=2000, seed=1)
    strat = build_strategy("momentum", {"breakout": 48})
    sig = strat.generate(df)
    assert list(sig.columns) == ["signal", "atr"]
    assert set(np.unique(sig["signal"].dropna())).issubset({-1, 0, 1})
    assert (sig["atr"].dropna() >= 0).all()
    assert len(sig) == len(df)


def test_momentum_long_only_has_no_shorts():
    df = generate_ohlcv(bars=2000, seed=2)
    sig = build_strategy("momentum", {"allow_short": False}).generate(df)
    assert (sig["signal"] >= 0).all()


def test_mean_reversion_output_contract():
    df = generate_ohlcv(bars=2000, seed=3)
    sig = build_strategy("mean_reversion", {}).generate(df)
    assert set(np.unique(sig["signal"].dropna())).issubset({-1, 0, 1})
    assert len(sig) == len(df)


def test_unknown_strategy_raises():
    try:
        build_strategy("does_not_exist")
        assert False, "expected ValueError"
    except ValueError:
        pass
