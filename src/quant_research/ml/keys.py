"""Row-level checks for ML datasets: unique keys and point-in-time feature dates (docs/ml-plan.md section 4)."""

import pandas as pd

from quant_research.ml.contracts import ContractError


def check_unique(df: pd.DataFrame, keys: tuple[str, ...] = ("date", "symbol")) -> None:
    """Refuse duplicate observation keys (no resolution rule is applied silently)."""
    dup = df.duplicated(list(keys), keep=False)
    if dup.any():
        sample = df.loc[dup, list(keys)].drop_duplicates().head(3).to_dict("records")
        raise ContractError(f"{int(dup.sum())} rows share a key {keys}; first: {sample}")


def check_point_in_time(df: pd.DataFrame, feature_date: str = "feature_date", decision: str = "date") -> None:
    """Refuse any row whose feature value is dated after its decision date, or has no date."""
    f, d = pd.to_datetime(df[feature_date]), pd.to_datetime(df[decision])
    bad = f.isna() | (f > d)
    if bad.any():
        raise ContractError(f"{int(bad.sum())} feature values are undated or dated after their decision date")
