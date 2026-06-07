# Automated Trading Bot (crypto, intraday)

A backtest-first framework for researching and (eventually) running an intraday
crypto trading bot. The design priority is **not lying to yourself**: realistic
costs, no lookahead, and out-of-sample evaluation baked in, so that any edge you
find is more likely to survive contact with a live market.

> **Status:** research/backtest stage. No live trading is wired up yet — by
> design. You earn the right to risk money by first proving an edge survives
> costs out-of-sample, then paper trading it.

## Read this first — expectations

- Most retail trading bots **lose money after costs**. This framework makes that
  easy to see rather than easy to hide.
- A genuinely robust strategy might return **~5–20% a year**. That is a *good*
  outcome. Anything promising 100%+ is curve-fit, hiding tail risk, or a scam.
- You cannot target a return directly. You can only build honest tooling, hunt
  for an edge, and size it so a bad streak doesn't ruin you. The return follows.
- The included strategies are **starting points, not money printers.** On the
  bundled synthetic data the momentum strategy *loses* — that is the engine
  working correctly, not a defect.

## What's here

```
src/botcore/
  data/        synthetic OHLCV generator + ccxt loader (lazy import) + parquet cache
  indicators/  ema, sma, atr, donchian channels, zscore (no TA-Lib needed)
  strategy/    Strategy interface + MomentumBreakout + MeanReversion
  risk/        ATR-stop, fixed-fractional position sizing, no-leverage cap
  backtest/    bar-by-bar engine: fees + slippage + intrabar stops, no lookahead
  metrics/     Sharpe, Sortino, max drawdown, Calmar, profit factor, exposure
run_backtest.py  config-driven runner; prints in-sample vs out-of-sample report
config.yaml      one place to set data source, strategy, params, costs, risk
tests/           pytest suite covering engine correctness and metrics
```

### Design choices that keep results honest

- **No lookahead.** A bar's signal is decided on its close and executed at the
  *next* bar's open (the engine acts on `signal[t-1]` at `open[t]`). Indicators
  use only causal (backward) windows.
- **Costs on every fill.** Per-side taker fee + slippage in basis points are
  charged on traded notional. This is what kills most paper-profitable systems.
- **Stops checked intrabar.** If price gaps through your stop at the open, you
  fill at the open (worse), not at the stop.
- **Equity-aware sizing.** Fixed fraction of equity risked per trade, with stop
  distance from ATR, so you trade smaller when volatility is high. No leverage
  by default.
- **Out-of-sample by default.** The runner reserves a holdout slice and reports
  it separately. If OOS is much worse than in-sample, you fit noise.

## Quickstart

```bash
pip install -r requirements.txt
python run_backtest.py            # runs on synthetic data (no network needed)
pytest -q                         # run the test suite
```

Edit `config.yaml` to change strategy, parameters, costs, or risk. To run on a
different strategy:

```yaml
strategy:
  name: mean_reversion
  params: { window: 48, entry_z: 2.0, exit_z: 0.3, allow_short: true }
```

## Using real market data

Synthetic data validates the *plumbing*, not an *edge*. To evaluate a strategy
for real, switch to live exchange data on a machine with network access:

```bash
pip install ccxt
```
```yaml
data:
  source: ccxt
  exchange: binance     # pick an exchange reachable + legal in your region
  symbol: BTC/USDT
  timeframe: 1h
  start: "2021-01-01"
  end: "2024-01-01"
```

Many exchanges geo-block cloud/datacenter IPs (HTTP 403). Run the fetch from
your own machine. Fetched data is cached to `data/cache/` as parquet.

## Roadmap to live (the safe order)

1. **Research** — backtest strategies on *real* data with these costs. Demand a
   positive, stable out-of-sample result with a tolerable max drawdown.
2. **Walk-forward** — re-fit/evaluate over rolling windows, not one split, to
   check the edge isn't a one-time fluke. *(next module to build)*
3. **Paper trade** — run the strategy live against the exchange in read-only /
   testnet mode for weeks. Reconcile fills and slippage vs the backtest.
4. **Small live capital** — only after the above, risk an amount you can fully
   afford to lose. Keep risk-per-trade small; let the equity curve, not hope,
   decide whether to scale.

## Not yet implemented (intentionally)

- Live/paper execution loop and exchange order routing
- Walk-forward optimisation and parameter search (overfitting-aware)
- Funding/borrow costs for perpetuals/margin

These come *after* you have an edge worth trading. Building execution before
edge is how people automate losing money faster.

## Disclaimer

Educational software. Trading carries substantial risk of loss. Nothing here is
financial advice. You are solely responsible for any capital you deploy.
