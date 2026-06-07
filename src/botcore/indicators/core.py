"""Lightweight technical indicators (no TA-Lib dependency).

All functions take/return pandas Series or DataFrames aligned to the input
index. They are deliberately simple and vectorised so they are easy to audit —
a hidden lookahead bug in an indicator silently inflates every backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rolling_high(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).max()


def rolling_low(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).min()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)
