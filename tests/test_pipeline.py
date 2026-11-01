"""Fast tests on synthetic data: no database or dataset needed."""
import numpy as np
import pandas as pd
import pytest

from olist_reviews.config import FEATURES, NUMERIC_FEATURES, TARGET, TIME_COLUMN
from olist_reviews.features import temporal_split
from olist_reviews.modeling import best_f1_threshold, evaluate, make_pipeline


def synthetic_orders(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    df["delay_days"] = rng.integers(-15, 15, n)
    df.loc[rng.random(n) < 0.05, "handling_days"] = np.nan  # trees must cope with gaps
    df["category"] = rng.choice(["toys", "books", "furniture", "audio"], n)
    df["customer_state"] = rng.choice(["SP", "RJ", "MG"], n)
    df["seller_state"] = rng.choice(["SP", "PR"], n)
    df["payment_type"] = rng.choice(["credit_card", "boleto"], n)
    risk = 1 / (1 + np.exp(-(0.3 * df["delay_days"] - 1.5)))
    df[TARGET] = (rng.random(n) < risk).astype(int)
    df[TIME_COLUMN] = pd.date_range("2017-01-01", periods=n, freq="h")
    return df


def test_temporal_split_is_strictly_ordered():
    df = synthetic_orders().sample(frac=1, random_state=1)
    train, test = temporal_split(df, 0.25)
    assert len(train) + len(test) == len(df)
    assert train[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    assert len(test) == pytest.approx(0.25 * len(df), abs=2)


@pytest.mark.parametrize("model_type", ["logreg", "rf", "xgb"])
def test_pipelines_learn_the_signal(model_type):
    train, test = temporal_split(synthetic_orders(1200))
    params = {"n_estimators": 50} if model_type != "logreg" else None
    pipe = make_pipeline(model_type, params).fit(train[FEATURES], train[TARGET])
    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    assert evaluate(test[TARGET], proba, 0.5)["roc_auc"] > 0.75


def test_unseen_categories_do_not_crash():
    df = synthetic_orders()
    pipe = make_pipeline("xgb", {"n_estimators": 20}).fit(df[FEATURES], df[TARGET])
    new = df[FEATURES].head(3).assign(category="brand_new_category", payment_type="voucher")
    assert pipe.predict_proba(new).shape == (3, 2)


def test_threshold_comes_from_training_data_only():
    df = synthetic_orders()
    t = best_f1_threshold(make_pipeline("logreg"), df[FEATURES], df[TARGET])
    assert 0 < t < 1


def test_evaluate_top_decile_and_base_rate():
    y = pd.Series([1] * 10 + [0] * 90)
    proba = np.r_[np.linspace(0.9, 0.8, 10), np.linspace(0.5, 0.0, 90)]
    m = evaluate(y, proba, 0.6)
    assert m["precision_top10"] == 1.0 and m["base_rate"] == 0.1 and m["f1"] == 1.0
