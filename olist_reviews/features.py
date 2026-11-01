"""Loading the modelling table and splitting it in time."""
import pandas as pd
from sqlalchemy import Engine

from olist_reviews.config import TEST_SHARE, TIME_COLUMN
from olist_reviews.db import get_engine


def load_features(engine: Engine | None = None) -> pd.DataFrame:
    """One row per delivered order from `features_view` (see sql/features_view.sql)."""
    df = pd.read_sql("SELECT * FROM features_view", engine or get_engine(), parse_dates=[TIME_COLUMN])
    return df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_share: float = TEST_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older purchases, test on the most recent ones.

    A random split lets the model learn from orders placed after the ones it is tested on, which
    hides drift (e.g. the 2018 truckers' strike delays) and overstates performance.
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1")
    ordered = df.sort_values(TIME_COLUMN, kind="stable")
    cutoff = ordered[TIME_COLUMN].iloc[int(len(ordered) * (1 - test_share))]
    train = ordered[ordered[TIME_COLUMN] < cutoff]
    test = ordered[ordered[TIME_COLUMN] >= cutoff]
    return train, test
