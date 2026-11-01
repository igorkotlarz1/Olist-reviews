"""Model pipelines, hyperparameter tuning and threshold selection.

Every preprocessing step lives inside a scikit-learn Pipeline, so it is re-fitted on the training
part of every CV fold. Target encoding fitted once on the whole training set and then
cross-validated leaks the validation labels into the features.
"""
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from olist_reviews.config import FEATURES, NUMERIC_FEATURES, ONE_HOT, SEED, TARGET_ENCODED

ModelType = Literal["logreg", "rf", "xgb"]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(model_type: ModelType) -> ColumnTransformer:
    """Trees need no scaling, log transforms or clipping (splits are invariant to monotone
    transforms) and handle missing values natively; the linear baseline needs all of that."""
    if model_type == "logreg":
        numeric = Pipeline([
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        numeric = "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            # cross-fitted in fit_transform: a row's own label never feeds its encoding
            ("target", TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=SEED)), TARGET_ENCODED),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONE_HOT),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_model(model_type: ModelType, params: dict | None = None, device: str = "cpu"):
    params = params or {}
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, C=params.get("C", 1.0))
    if model_type == "rf":
        defaults = {"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.5, "max_samples": 0.5,
                        "n_jobs": -1, "random_state": SEED}
        return RandomForestClassifier(**{**defaults, **params})
    if model_type == "xgb":
        defaults = {"tree_method": "hist", "device": device, "n_jobs": -1, "random_state": SEED, "eval_metric": "aucpr"}
        return XGBClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model type: {model_type}")


def make_pipeline(model_type: ModelType, params: dict | None = None, device: str = "cpu") -> Pipeline:
    return Pipeline([("prep", make_preprocessor(model_type)), ("model", make_model(model_type, params, device))])


def xgb_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
    }


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_xgb(X: pd.DataFrame, y: pd.Series, n_trials: int = 40, device: str = "cpu") -> optuna.Study:
    """Maximise cross-validated average precision (PR-AUC) of the full pipeline.

    PR-AUC is threshold-free and focuses on the minority class; the decision threshold is chosen
    separately, afterwards. No class weights: the model outputs real probabilities, which the
    app shows as the risk of a bad review.
    """
    def objective(trial: optuna.Trial) -> float:
        pipe = make_pipeline("xgb", xgb_search_space(trial), device)
        return cross_val_score(pipe, X[FEATURES], y, cv=cv_splitter(), scoring="average_precision").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="xgb_review_risk")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def best_f1_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Pick the threshold on out-of-fold training predictions, never on the test set."""
    oof = cross_val_predict(pipe, X[FEATURES], y, cv=cv_splitter(), method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = proba >= threshold
    top = np.argsort(-proba)[: max(1, len(proba) // 10)]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "threshold": threshold,
        # share of bad reviews among the 10% riskiest orders, vs the base rate
        "precision_top10": float(np.asarray(y)[top].mean()),
        "base_rate": float(np.mean(y)),
    }
