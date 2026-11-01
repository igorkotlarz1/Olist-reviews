"""Paths and feature definitions shared by the ETL, training and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "review_risk.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

SEED = 123
TARGET = "bad_review"
TIME_COLUMN = "purchase_ts"
TEST_SHARE = 0.2  # the most recent 20% of purchases form the hold-out set

NUMERIC_FEATURES = [
    "num_items", "num_products", "num_sellers",
    "total_price", "total_freight", "freight_ratio",
    "num_payments", "installments",
    "name_len", "desc_len", "photos", "weight_g", "volume_l",
    "same_state",
    "seller_prior_reviews", "seller_prior_bad_rate", "product_prior_reviews", "product_prior_bad_rate",
    "promised_days", "approval_hours", "handling_days", "seller_delay_days",
    "transit_days", "delivery_days", "delay_days", "is_late",
]
# high-cardinality columns get target encoding, the rest one-hot
TARGET_ENCODED = ["category", "customer_state", "seller_state"]
ONE_HOT = ["payment_type"]
CATEGORICAL_FEATURES = TARGET_ENCODED + ONE_HOT
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
