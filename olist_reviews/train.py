"""Train and evaluate the review-risk models, then save the best pipeline for the app.

    python -m olist_reviews.train [--trials 40] [--device cuda]

Protocol: temporal hold-out (latest 20% of purchases) -> Optuna tunes XGBoost with 5-fold CV on
the training part -> decision threshold from out-of-fold predictions -> one final look at the
hold-out set. A logistic regression and a random forest are trained the same way for reference.
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
import xgboost

from olist_reviews.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
    TIME_COLUMN,
)
from olist_reviews.features import load_features, temporal_split
from olist_reviews.modeling import best_f1_threshold, evaluate, make_pipeline, tune_xgb


def app_defaults(train: pd.DataFrame) -> dict:
    """Typical values for every feature, so the app can fill in what the user does not set."""
    numeric = train[NUMERIC_FEATURES].median().round(2).to_dict()
    categorical = {c: train[c].mode().iloc[0] for c in CATEGORICAL_FEATURES}
    options = {c: train[c].value_counts().index.tolist() for c in CATEGORICAL_FEATURES}
    return {"values": {**numeric, **categorical}, "options": options}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials for XGBoost")
    parser.add_argument("--device", default="cpu", help="XGBoost device: cpu or cuda")
    args = parser.parse_args()

    df = load_features()
    train, test = temporal_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"train: {len(train):,} orders up to {train[TIME_COLUMN].max():%Y-%m-%d} | "
          f"test: {len(test):,} orders from {test[TIME_COLUMN].min():%Y-%m-%d}")

    study = tune_xgb(X_train, y_train, n_trials=args.trials, device=args.device)
    print(f"best CV PR-AUC {study.best_value:.3f} with {study.best_params}")

    results, pipelines = {}, {}
    for name, params in [("logreg", None), ("rf", None), ("xgb", study.best_params)]:
        pipe = make_pipeline(name, params, args.device)
        threshold = best_f1_threshold(pipe, X_train, y_train)
        pipe.fit(X_train, y_train)
        results[name] = evaluate(y_test, pipe.predict_proba(X_test)[:, 1], threshold)
        pipelines[name] = pipe
        print(f"{name:7s} " + "  ".join(f"{k}={v:.3f}" for k, v in results[name].items()))

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {
            "train_orders": len(train), "test_orders": len(test),
            "test_from": f"{test[TIME_COLUMN].min():%Y-%m-%d}", "test_to": f"{test[TIME_COLUMN].max():%Y-%m-%d}",
        },
        "xgb_cv_pr_auc": study.best_value,
        "xgb_params": study.best_params,
        "test": results,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipelines["xgb"],
        "threshold": results["xgb"]["threshold"],
        "features": FEATURES,
        "defaults": app_defaults(train),
        "metrics": results["xgb"],
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }, MODEL_PATH, compress=3)
    print(f"saved {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
