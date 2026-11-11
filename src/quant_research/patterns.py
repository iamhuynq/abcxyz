"""Pattern library: named, versioned conditions over `feature_target` (doc sections 5, 13, 14, 24, 28).

The version is a hash of the condition and its base, so editing a condition automatically creates a
new version and old results stay attributable to the old definition.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    name: str
    where: str
    hypothesis: str
    doc_ref: str = ""
    base: str | None = None     # a simpler pattern this one refines (doc 5.2: does the extra condition add information?)
    decile_of: str | None = None  # optional: keep only this cross-sectional decile of a feature, per date
    decile: int = 10

    @property
    def version(self) -> str:
        key = f"{self.where}|{self.base}|{self.decile_of}|{self.decile if self.decile_of else ''}"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def source_sql(self, universe_sql: str, min_stocks_per_date: int, table: str = "rs.feature_target",
                   row_filter: str = "period = ?") -> str:
        """Rows of `table` in the filtered universe matching this pattern; `row_filter` binds one parameter.

        Research uses feature_target + a period; the daily scanner uses stock_features (no target columns,
        so it cannot read the future) + a single date.
        """
        if not self.decile_of:
            return f"SELECT * FROM {table} WHERE {universe_sql} AND {row_filter} AND ({self.where})"
        f = self.decile_of
        return f"""SELECT * EXCLUDE (_decile, _n) FROM (
            SELECT *, ntile(10) OVER (PARTITION BY date ORDER BY {f}, symbol) AS _decile,   -- ties: by symbol
                   count(*) OVER (PARTITION BY date) AS _n
            FROM {table}
            WHERE {universe_sql} AND {row_filter} AND {f} IS NOT NULL AND isfinite({f}))
            WHERE _decile = {self.decile} AND _n >= {min_stocks_per_date} AND ({self.where})"""


_DROP3 = "return_1d < -0.03"
_DROP3_VOL2 = f"{_DROP3} AND volume_ratio_20 > 2"

LIBRARY: dict[str, Pattern] = {p.name: p for p in (
    Pattern("drop3", _DROP3, "A one-day drop of more than 3%: what follows?", "5.1"),
    Pattern("drop3_volume2", _DROP3_VOL2,
            "A sharp drop on heavy volume behaves differently from a quiet drop", "5.2", base="drop3"),
    Pattern("drop3_volume2_sellers", f"{_DROP3_VOL2} AND order_imbalance < -0.4",
            "Seller-dominated order book adds information to a heavy-volume drop", "5.2", base="drop3_volume2"),
    Pattern("drop3_volume2_foreign_sell", f"{_DROP3_VOL2} AND foreign_net_value < 0 AND order_imbalance < -0.3",
            "Drop + volume + foreign selling + seller imbalance (doc 28 example)", "28", base="drop3_volume2"),
    Pattern("surge4_volume_close_high", "return_1d > 0.04 AND volume_zscore_20 > 2 AND close_position > 0.8",
            "Strong up day on unusual volume closing near the high", "28"),
    Pattern("volume_spike", "volume_ratio_20 > 3", "Traded value above 3x its 20-session average", "24"),
    Pattern("volume_spike_flat", "volume_ratio_20 > 8 AND abs(return_1d) < 0.002",
            "Huge volume while the price barely moves (anomaly)", "14", base="volume_spike"),
    Pattern("foreign_accumulation_no_breakout", "foreign_net_ratio_20 > 0.1 AND return_20d < 0.02",
            "Foreign investors accumulate while the price has not broken out", "13, 28"),
    Pattern("foreign_divergence_up", "return_5d > 0.05 AND foreign_net_5d < 0",
            "Price up while foreign investors sell (divergence)", "13"),
    Pattern("imbalance_positive_price_down", "order_imbalance > 0.5 AND return_1d < -0.02",
            "Buy-side order imbalance while the price falls (anomaly)", "14"),
    # Candidates selected from the research-period scan (2026-10-04); pre-registered for validation.
    Pattern("order_imbalance_d10", "TRUE", "Top decile of order imbalance outperforms (long side of the scan spread)",
            "2.2", decile_of="order_imbalance"),
    Pattern("order_imbalance_d10_no_limit", "NOT limit_up",
            "The order-imbalance effect is not just unfilled buy queues at the ceiling", "2.2",
            base="order_imbalance_d10", decile_of="order_imbalance"),
    Pattern("momentum10_d10", "TRUE", "Top decile of 10-session return keeps outperforming (short-term momentum)",
            "3.1", decile_of="return_10d"),
    Pattern("momentum5", "return_5d > 0.05", "A 5-session rise above 5% keeps outperforming", "3.1"),
    Pattern("momentum5_foreign_sell", "return_5d > 0.05 AND foreign_net_5d < 0",
            "Foreign selling adds information to a 5-session rise (decomposes foreign_divergence_up)", "13",
            base="momentum5"),
)}
