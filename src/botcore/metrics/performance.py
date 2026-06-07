"""Performance metrics for an equity curve / return series.

Annualised stats need bars-per-year (periods_per_year) because we work on
intraday bars. Sharpe/Sortino here assume a zero risk-free rate (fine for
relative comparison; adjust if you care about absolute risk-adjusted return).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    num_trades: int
    exposure: float
    final_equity: float

    def as_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    position: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
) -> Metrics:
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0

    n = len(returns)
    years = n / periods_per_year if periods_per_year else np.nan
    cagr = (final / initial) ** (1 / years) - 1 if years and years > 0 and final > 0 else np.nan

    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    mean_ret = float(returns.mean())
    sharpe = (mean_ret * periods_per_year) / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=0) * np.sqrt(periods_per_year)) if len(downside) else 0.0
    sortino = (mean_ret * periods_per_year) / downside_dev if downside_dev > 0 else 0.0

    max_dd = _max_drawdown(equity)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 and not np.isnan(cagr) else np.nan

    # Trade-level stats from realised per-round-trip PnL.
    win_rate, profit_factor, num_trades = _trade_stats(trades)

    exposure = float((position != 0).mean())

    return Metrics(
        total_return=total_return, cagr=cagr, ann_volatility=ann_vol,
        sharpe=sharpe, sortino=sortino, max_drawdown=max_dd, calmar=calmar,
        win_rate=win_rate, profit_factor=profit_factor, num_trades=num_trades,
        exposure=exposure, final_equity=final,
    )


def _trade_stats(trades: pd.DataFrame) -> tuple[float, float, int]:
    """Reconstruct round-trip PnL from the fill blotter (cash-flow based).

    Each fill changes cash by -delta_units*price - fee. A round trip is the
    sequence of fills between flat states; its PnL is the net cash change over
    that sequence.
    """
    if trades is None or trades.empty:
        return 0.0, 0.0, 0

    pnls: list[float] = []
    cash_flow = 0.0
    units = 0.0
    for _, row in trades.iterrows():
        cash_flow += -row["delta_units"] * row["price"] - row["fee"]
        units += row["delta_units"]
        if abs(units) < 1e-12:  # back to flat => close round trip
            pnls.append(cash_flow)
            cash_flow = 0.0
            units = 0.0

    if not pnls:
        return 0.0, 0.0, 0

    pnls_arr = np.array(pnls)
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]
    win_rate = float(len(wins) / len(pnls_arr))
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    return win_rate, profit_factor, len(pnls_arr)
