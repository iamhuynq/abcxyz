-- Outcomes after session t using data > t ONLY (doc section 4). Kept apart from stock_features.
-- Two conventions: close-to-close from t (the doc's statistics), and executable from the next open
-- (enter at open t+1, exit at close t+h; h >= 3 respects T+2 settlement). This file must never
-- look backward (enforced by a test).
CREATE OR REPLACE TABLE stock_targets AS
WITH f AS (
    SELECT p.symbol, p.date, p.period, p.adj_close,
        lead(date, 1)  OVER w AS d1,  lead(date, 3)  OVER w AS d3,  lead(date, 5)  OVER w AS d5,
        lead(date, 10) OVER w AS d10, lead(date, 20) OVER w AS d20,
        lead(adj_open, 1)   OVER w AS open_1,
        lead(adj_close, 1)  OVER w AS close_1,
        lead(adj_close, 3)  OVER w AS close_3,
        lead(adj_close, 5)  OVER w AS close_5,
        lead(adj_close, 10) OVER w AS close_10,
        lead(adj_close, 20) OVER w AS close_20,
        lead(open_limit_up, 1) OVER w AS next_open_limit_up,
        coalesce(bool_or(price_jump) OVER (w ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING), false)  AS jump_ahead_1,
        coalesce(bool_or(price_jump) OVER (w ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING), false)  AS jump_ahead_3,
        coalesce(bool_or(price_jump) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING), false)  AS jump_ahead_5,
        coalesce(bool_or(price_jump) OVER (w ROWS BETWEEN 1 FOLLOWING AND 10 FOLLOWING), false) AS jump_ahead_10,
        coalesce(bool_or(price_jump) OVER (w ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING), false) AS jump_ahead_20,
        CASE WHEN count(adj_high) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) = 5
             THEN max(adj_high) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) END AS max_high_5,
        CASE WHEN count(adj_low) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) = 5
             THEN min(adj_low) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) END AS min_low_5
    FROM daily_panel p
    WINDOW w AS (PARTITION BY symbol ORDER BY date)
)
SELECT
    f.symbol, f.date, f.period,
    -- close-to-close (doc): adj_close_{t+h} / adj_close_t - 1
    f.close_1  / f.adj_close - 1 AS fwd_ret_close_1d,
    f.close_3  / f.adj_close - 1 AS fwd_ret_close_3d,
    f.close_5  / f.adj_close - 1 AS fwd_ret_close_5d,
    f.close_10 / f.adj_close - 1 AS fwd_ret_close_10d,
    -- executable: enter next open, exit close t+h
    f.close_3  / f.open_1 - 1 AS fwd_ret_exec_3d,
    f.close_5  / f.open_1 - 1 AS fwd_ret_exec_5d,
    f.close_10 / f.open_1 - 1 AS fwd_ret_exec_10d,
    f.close_20 / f.open_1 - 1 AS fwd_ret_exec_20d,
    -- executable minus VNINDEX over the same span (open of d1 to close of dh)
    (f.close_5  / f.open_1 - 1) - (m5.mkt_close  / m1.mkt_open - 1) AS fwd_excess_exec_5d,
    (f.close_10 / f.open_1 - 1) - (m10.mkt_close / m1.mkt_open - 1) AS fwd_excess_exec_10d,
    (f.close_20 / f.open_1 - 1) - (m20.mkt_close / m1.mkt_open - 1) AS fwd_excess_exec_20d,
    f.max_high_5 / f.open_1 - 1 AS fwd_max_return_5d,
    f.min_low_5  / f.open_1 - 1 AS fwd_max_drawdown_5d,
    CASE WHEN f.close_5 IS NOT NULL THEN f.close_5 / f.adj_close - 1 > 0.03 END AS y_up3_5d,
    f.next_open_limit_up AS entry_blocked,
    f.jump_ahead_20      AS fwd_has_price_jump_20d,
    f.d20 IS NOT NULL    AS target_complete,
    CASE WHEN f.d20 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d20 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d20 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d20 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period,
    -- Per-horizon outcome masks (review of 2026-11-10): the same masks over the target's own window.
    -- The two columns above keep the legacy 20-session window for recorded definitions.
    f.jump_ahead_1  AS fwd_has_price_jump_1d,
    f.jump_ahead_3  AS fwd_has_price_jump_3d,
    f.jump_ahead_5  AS fwd_has_price_jump_5d,
    f.jump_ahead_10 AS fwd_has_price_jump_10d,
    CASE WHEN f.d1 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d1 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d1 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d1 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period_1d,
    CASE WHEN f.d3 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d3 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d3 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d3 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period_3d,
    CASE WHEN f.d5 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d5 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d5 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d5 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period_5d,
    CASE WHEN f.d10 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d10 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d10 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d10 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period_10d,
    CASE WHEN f.d20 IS NULL THEN NULL
         ELSE f.period <> CASE WHEN f.d20 <= DATE '2023-12-31' THEN 'research'
                               WHEN f.d20 <= DATE '2025-12-31' THEN 'validation'
                               WHEN f.d20 <= DATE '2026-10-02' THEN 'holdout'
                               ELSE 'forward' END END AS crosses_period_20d
FROM f
LEFT JOIN market_daily m1  ON m1.date  = f.d1
LEFT JOIN market_daily m5  ON m5.date  = f.d5
LEFT JOIN market_daily m10 ON m10.date = f.d10
LEFT JOIN market_daily m20 ON m20.date = f.d20;
