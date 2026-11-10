# Feature Dictionary - research.duckdb (feature set v1)

Built by `uv run quant build` from `data/warehouse.duckdb` (attached read-only). Plan:
`docs/feature-engine-plan.md`. Latest sanity report: `data/reports/feature-sanity-{date}.md`.
Parquet: `data/parquet/research/{daily_panel,market_daily,stock_features,stock_targets}.parquet`.

```python
import duckdb
con = duckdb.connect("data/research.duckdb", read_only=True)
df = con.sql("""
    SELECT * FROM feature_target
    WHERE period = 'research' AND is_traded AND adv_value_20 > 1e9
      AND NOT price_jump AND NOT fwd_has_price_jump_20d AND NOT bad_source_date
""").df()
```

## Recommended default filters

| Filter | Why |
|--------|-----|
| `is_traded` | 33% of panel rows are no-trade sessions (illiquid stocks): price is the reference, volume 0 |
| `adv_value_20 > 1e9` (VND) | liquidity; ~1.2M of 5.2M rows pass at 1 bn VND/day |
| `NOT price_jump`, `NOT fwd_has_price_jump_20d` | adjusted one-day moves > 40% are source errors (737 rows) |
| `NOT bad_source_date` | 2025-07-16 volume corruption |
| `NOT foreign_inconsistent` when using foreign features | foreign value > total value on 3,584 rows |
| `period = 'research'` | <= 2023; `validation` 2024-2025; `holdout` 2026+ (do not look at holdout) |
| `NOT crosses_period` | target window spills into the next period |

Point-in-time universe and outcome masks (review of 2026-11-10). `ResearchParams.point_in_time_sql()` is
the universe known at the close of t (`is_traded`, ADV, age, `NOT price_jump`, `NOT bad_source_date`).
Outcome masks are separate and named:

| Mask | Column (legacy 20-session window) | Per-horizon column (h = 1, 3, 5, 10, 20) |
|------|-----------------------------------|-------------------------------------------|
| `quality` | `fwd_has_price_jump_20d` | `fwd_has_price_jump_{h}d` |
| `execution` | `entry_blocked` | same (the next open) |
| `period` | `crosses_period` | `crosses_period_{h}d` |

`ResearchParams.sample_sql(target)` = universe AND the chosen masks. The defaults (all three masks, legacy
window) are the recorded definition used by every logged run; `mask_window="horizon"` is for new
definitions.

## Conventions

- t = the session of the row. **Features use data <= t; targets use data > t** (separate tables).
- Windows count the symbol's own sessions. A window value is NULL until the window is full.
- Prices: `adj_* = raw / adj_ratio` (back-adjusted for dividends, stock dividends, rights). Returns and
  ratios use adjusted prices; `close_raw` is for absolute price filters.
- Volume features use **matched value in VND** (`deal_value = total_value - putthrough_value`), not
  shares, so stock dividends and splits do not create fake spikes. Put-through (block) trades excluded.
- Zero denominators give NULL. No infinite values (checked by the sanity report).

## stock_features

| Feature | Definition | Notes |
|---------|------------|-------|
| `return_1d/3d/5d/10d/20d` | `adj_close_t / adj_close_{t-n} - 1` | |
| `excess_return_5d/20d` | stock return minus VNINDEX return over n sessions | approximate if the stock had missing sessions |
| `volume_ratio_5/20` | `deal_value_t / mean(deal_value_{t-n..t-1})` | baseline excludes today |
| `volume_change` | `deal_value_t / deal_value_{t-1} - 1` | NULL after a no-trade day |
| `volume_zscore_20` | z-score of `ln(1 + deal_value_t)` vs previous 20 sessions | |
| `volatility_5/20` | sample stdev of `return_1d`, window incl. t | |
| `atr_14_pct` | mean true range (14, adjusted) / `adj_close_t` | first session has no true range |
| `high_low_range` | `(high - low) / close` | |
| `gap` | `adj_open_t / adj_close_{t-1} - 1` | |
| `close_position` | `(close - low) / (high - low)` | NULL when high = low (incl. all no-trade days) |
| `close_vs_avg_price` | `(close_raw - vwap_deal) / vwap_deal`, `vwap_deal = deal_value / (deal_volume * 1000)` | exact matched VWAP, not the rounded `price_average` |
| `order_imbalance` | `(buy_qty - sell_qty) / (buy_qty + sell_qty)` | **order placements**, not fills; NULL where the source has none |
| `volume_imbalance` | `(buy_count - sell_count) / (buy_count + sell_count)` | order counts; sparse before 2010 |
| `buy_pressure`, `sell_pressure` | `buy_qty / deal_volume`, `sell_qty / deal_volume` | usually > 1 (placed > matched); heavy tails |
| `buy_sell_ratio` | `buy_qty / sell_qty` | heavy tails |
| `foreign_net_volume`, `foreign_net_value` | foreign buy - sell (shares, VND) | source has impossible extremes; see flag |
| `foreign_net_3d/5d/20d` | rolling sum of `foreign_net_value` | |
| `foreign_net_ratio_20` | `foreign_net_20d / sum(deal_value, 20)` | size-normalised |
| `foreign_intensity` | `(foreign buy + sell value) / total_value` | > 1 means inconsistent source row |
| `market_regime` | VNINDEX: Bull (close > MA200 and MA50 > MA200), Bear (both below), else Sideway | NULL for the first 199 sessions of 2000 |
| `trend_regime` | VNINDEX above/below MA200 (the doc's simple rule) | |
| `vol_regime` | VNINDEX 20-session vol above/below its trailing 250-session median | |
| `adv_value_20` | mean `deal_value` over 20 sessions (VND) | liquidity filter |

Flags: `is_traded`, `limit_up`, `limit_down` (approximate, see below), `bad_source_date`,
`session_index` (sessions since first trade; exclude the first ~20 for IPO effects), `price_jump`,
`foreign_inconsistent`.

## stock_targets

| Target | Definition |
|--------|------------|
| `fwd_ret_close_1d/3d/5d/10d` | `adj_close_{t+h} / adj_close_t - 1` (doc section 4; statistics only) |
| `fwd_ret_exec_3d/5d/10d/20d` | `adj_close_{t+h} / adj_open_{t+1} - 1`: buy next open, sell close t+h (T+2-compliant) |
| `fwd_excess_exec_5d/10d/20d` | exec return minus VNINDEX over the same span (open d1 to close dh) |
| `fwd_max_return_5d`, `fwd_max_drawdown_5d` | max high / min low over t+1..t+5 vs next open |
| `y_up3_5d` | `fwd_ret_close_5d > 3%` (doc classification target) |
| `entry_blocked` | t+1 opens at the (approximate) ceiling: the buy is unrealistic |
| `fwd_has_price_jump_20d` | a `price_jump` occurs in t+1..t+20: exclude |
| `target_complete`, `crosses_period` | 20 future sessions exist; target window spills into the next period |
| `fwd_has_price_jump_{h}d`, `crosses_period_{h}d` | the same masks over the target's own window, h = 1, 3, 5, 10 (and `crosses_period_20d`) |

## stock_factors (Factor Engine, factor set f1)

Built by `quant build` (`quant_research/factors.py`, definitions also in the `factor_definitions` table). One row
per (date, symbol, factor) for the scoring universe: `is_traded`, `adv_value_20 > 1e9`, `session_index > 20`,
`NOT price_jump`, `NOT bad_source_date`, and at least 30 scored stocks per date and factor.

| Column | Definition |
|--------|------------|
| `value` | raw factor value (trailing data only) |
| `winsorized` | `value` clipped to the 1st / 99th percentile of the date |
| `rank_pct` | percentile rank of `value` on the date: 0 = lowest, 1 = highest (ties share the lowest rank) |
| `z` | z-score of `winsorized` on the date (population standard deviation) |
| `bucket` | quintile from `rank_pct`, 1 to 5 |
| `z_industry` | `z` minus the mean `z` of the stock's ICB level-2 industry that date (at least 3 stocks) |

Factors: `liquidity` (ln ADV20, also the size proxy), `momentum_12_1`, `momentum_1m`, `reversal_1w`,
`volatility`, `beta` (250 sessions), `order_flow` (5-session mean order imbalance), `foreign_flow`,
`volume_surge`. Structure report: `uv run quant factors describe`.

## market_regimes (Regime Engine)

One row per market session (`quant_research/regimes.py`). Every label uses data up to that session only.
Continuous dimensions are labelled by the tercile of today's value within the trailing 500 sessions (today
included; NULL before 250 values). `*_value` and `*_pct` columns hold the underlying value and its
percentile.

| Dimension | Underlying value | States |
|-----------|------------------|--------|
| `direction` | `market_daily.market_regime` | Bear / Sideway / Bull |
| `volatility` | VNINDEX 20-session volatility | low / normal / high |
| `liquidity` | total `deal_value`, 20-session mean | low / normal / high |
| `breadth` | share of liquid stocks above their own 50-session mean | weak / neutral / strong |
| `foreign` | market foreign net value / turnover, 20 sessions (quote it: SQL keyword) | selling / neutral / buying |
| `risk` | risk_off: volatility high and breadth weak, or `drawdown` <= -20%; risk_on: volatility not high, breadth strong, direction Bull | risk_off / neutral / risk_on |

Trending series (liquidity, foreign flow) sit in their upper tercile more often than in the lower one.

## stock_trading_costs (cost model v1 inputs)

One row per (symbol, date) (`quant_research/backtest/costs.py`). The backtest engine reads the row of the
previous close.

| Column | Definition |
|--------|------------|
| `tick_half_spread` | half a tick / raw close (HOSE 10 / 50 / 100 VND by price, HNX / UPCOM 100 VND; today's exchange) |
| `chl_half_spread` | Abdi-Ranaldo (2017) close-high-low estimate over the pairs in the last 21 sessions (k + 1 <= t), half of the spread, negative means floored at 0, NULL under 10 pairs |
| `half_spread` | max of the two (overstates liquid stocks: CHL is noisy) |
| `sigma_20`, `adv_value_20` | inputs of the square-root impact `k * sigma_20 * sqrt(order / ADV20)` |

## Known limitations

- **Exchange history is unknown.** `limit_up/limit_down/entry_blocked` use today's exchange band
  (HOSE 7%, HNX 10%, UPCOM 15%) and today's tick rules; they are wrong for periods when a stock traded
  on another exchange or when bands differed (e.g. 2008). Treat them as approximate.
- Industry classification is current only.
- Order statistics are placements (can include cancelled orders) and are sparse before ~2010.
- Source extremes remain in foreign and order-flow features (heavy tails): winsorize or rank-transform
  before pooled statistics.
