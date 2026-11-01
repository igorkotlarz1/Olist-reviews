"""Extract the Kaggle CSVs, clean them lightly and load them into PostgreSQL.

Run with `python -m olist_reviews.etl`. Each run drops and recreates the schema, so it can be
repeated safely. Everything happens in one transaction: a failure leaves the database untouched.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Connection

from olist_reviews.config import DATA_DIR, SQL_DIR
from olist_reviews.db import get_engine

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", format=DATE_FORMAT)
    return df


def transform_customers(df: pd.DataFrame) -> pd.DataFrame:
    df["customer_city"] = df["customer_city"].str.lower()
    return df


def transform_sellers(df: pd.DataFrame) -> pd.DataFrame:
    df["seller_city"] = df["seller_city"].str.lower()
    return df


def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    return df


def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
    return _parse_dates(df, [
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ])


def transform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["review_comment_title", "review_comment_message"]:
        df[col] = df[col].fillna("").str.replace(r"[\n\r]+", " ", regex=True).str.strip()
    return _parse_dates(df, ["review_creation_date", "review_answer_timestamp"])


def transform_order_items(df: pd.DataFrame) -> pd.DataFrame:
    return _parse_dates(df, ["shipping_limit_date"])


def identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


@dataclass(frozen=True)
class Table:
    name: str
    csv: str
    transform: Callable[[pd.DataFrame], pd.DataFrame] = identity
    dtypes: dict[str, str] = field(default_factory=dict)


# Order matters: parents before the tables that reference them.
TABLES = [
    Table("customers", "olist_customers_dataset.csv", transform_customers,
          {"customer_zip_code_prefix": "str"}),
    Table("sellers", "olist_sellers_dataset.csv", transform_sellers, {"seller_zip_code_prefix": "str"}),
    Table("products", "olist_products_dataset.csv", transform_products,
          {c: "Int64" for c in ["product_name_lenght", "product_description_lenght", "product_photos_qty",
                                "product_weight_g", "product_length_cm", "product_height_cm",
                                "product_width_cm"]}),
    Table("category_names", "product_category_name_translation.csv"),
    Table("orders", "olist_orders_dataset.csv", transform_orders),
    Table("reviews", "olist_order_reviews_dataset.csv", transform_reviews),
    Table("order_items", "olist_order_items_dataset.csv", transform_order_items),
    Table("order_payments", "olist_order_payments_dataset.csv"),
]


def load_table(conn: Connection, table: Table) -> int:
    path = DATA_DIR / table.csv
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Download the dataset into data/ (see README).")
    df = table.transform(pd.read_csv(path, dtype=table.dtypes or None))
    df.to_sql(table.name, conn, if_exists="append", index=False, chunksize=5000, method="multi")
    return len(df)


def run_sql(conn: Connection, filename: str) -> None:
    conn.exec_driver_sql((SQL_DIR / filename).read_text(encoding="utf-8"))


def main() -> None:
    with get_engine().begin() as conn:
        run_sql(conn, "schema.sql")
        for table in TABLES:
            print(f"{table.name:<15} {load_table(conn, table):>8,} rows")
        run_sql(conn, "features_view.sql")
        n = conn.exec_driver_sql("SELECT COUNT(*) FROM features_view").scalar_one()
    print(f"features_view   {n:>8,} rows")


if __name__ == "__main__":
    main()
