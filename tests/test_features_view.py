"""Leakage checks against the real database. Skipped when PostgreSQL or the data is not available."""
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError

from olist_reviews.db import get_engine


@pytest.fixture(scope="module")
def engine():
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1 FROM features_view LIMIT 1")
    except (OperationalError, ProgrammingError):
        pytest.skip("features_view not available (run the ETL first)")
    return eng


def test_one_row_per_order(engine):
    n, distinct = pd.read_sql("SELECT COUNT(*), COUNT(DISTINCT order_id) FROM features_view", engine).iloc[0]
    assert n == distinct


def test_no_review_answered_before_delivery(engine):
    bad = pd.read_sql("""
        SELECT COUNT(*) FROM features_view v
        JOIN orders o USING (order_id)
        JOIN reviews r ON r.order_id = v.order_id
        WHERE r.review_answer_timestamp < o.order_delivered_customer_date
          AND r.review_answer_timestamp = (SELECT MAX(review_answer_timestamp) FROM reviews WHERE order_id = v.order_id)
    """, engine).iloc[0, 0]
    assert bad == 0


def test_seller_history_uses_only_past_reviews(engine):
    """Recompute the seller reputation of a sample of orders by brute force."""
    sample = pd.read_sql("""
        SELECT v.order_id, v.seller_prior_reviews, v.seller_prior_bad_rate,
               o.order_delivered_customer_date::DATE AS delivered_on,
               (SELECT oi.seller_id FROM order_items oi WHERE oi.order_id = v.order_id
                ORDER BY oi.price DESC, oi.order_item_id LIMIT 1) AS seller_id
        FROM features_view v JOIN orders o USING (order_id)
        ORDER BY md5(v.order_id) LIMIT 200
    """, engine)
    for row in sample.itertuples():
        past = pd.read_sql("""
            SELECT DISTINCT r.review_answer_timestamp::DATE AS day, (r.review_score <= 3)::INT AS bad, r.order_id, r.review_id
            FROM reviews r JOIN order_items oi ON oi.order_id = r.order_id
            WHERE oi.seller_id = %(seller)s AND r.review_answer_timestamp::DATE < %(day)s
        """, engine, params={"seller": row.seller_id, "day": row.delivered_on})
        assert row.seller_prior_reviews == len(past)
        if len(past):
            assert row.seller_prior_bad_rate == pytest.approx(past["bad"].mean())
