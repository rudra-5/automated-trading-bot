"""Position sizing and stop placement.

The default model is *fixed-fractional risk with an ATR stop*: on each entry we
risk a fixed fraction of current equity, and the distance to the stop is a
multiple of ATR. Position size = risk_amount / stop_distance. This automatically
trades smaller when volatility is high and larger when it's calm, which is the
single most important thing separating survivors from blow-ups.

No leverage by default: size is capped so notional never exceeds equity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01     # fraction of equity risked per trade
    atr_stop_mult: float = 3.0       # stop distance = atr_stop_mult * ATR
    max_leverage: float = 1.0        # 1.0 => no leverage (notional <= equity)


def size_position(
    equity: float,
    price: float,
    atr_value: float,
    direction: int,
    cfg: RiskConfig,
) -> tuple[float, float]:
    """Return (signed_units, stop_price) for a new entry.

    direction: +1 long, -1 short. Returns (0, nan) if inputs are unusable.
    """
    if direction == 0 or price <= 0 or atr_value is None or atr_value <= 0:
        return 0.0, float("nan")

    stop_distance = cfg.atr_stop_mult * atr_value
    if stop_distance <= 0:
        return 0.0, float("nan")

    risk_amount = equity * cfg.risk_per_trade
    units = risk_amount / stop_distance

    # Cap by leverage limit (no leverage => notional <= equity).
    max_units = (equity * cfg.max_leverage) / price
    units = min(units, max_units)

    signed_units = units * direction
    stop_price = price - direction * stop_distance
    return signed_units, stop_price
