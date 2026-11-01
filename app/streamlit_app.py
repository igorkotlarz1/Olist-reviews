"""Streamlit dashboard: estimate the risk that a delivered Olist order gets a 1-3 star review.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from olist_reviews.config import MODEL_PATH

LABELS = {
    "num_items": "items in the order", "num_products": "distinct products", "num_sellers": "sellers in the order",
    "total_price": "order value", "total_freight": "freight cost", "freight_ratio": "freight / price",
    "num_payments": "payment methods", "installments": "installments", "name_len": "product name length",
    "desc_len": "description length", "photos": "product photos", "weight_g": "product weight",
    "volume_l": "product volume", "same_state": "seller in the customer's state",
    "seller_prior_reviews": "seller's review count", "seller_prior_bad_rate": "seller's bad-review rate",
    "product_prior_reviews": "product's review count", "product_prior_bad_rate": "product's bad-review rate",
    "promised_days": "promised delivery time", "approval_hours": "payment approval time",
    "handling_days": "seller handling time", "seller_delay_days": "seller dispatch delay",
    "transit_days": "carrier transit time", "delivery_days": "delivery time", "delay_days": "days late",
    "is_late": "late delivery", "category": "product category", "customer_state": "customer state",
    "seller_state": "seller state",
}

st.set_page_config(page_title="Olist review risk", page_icon="📦", layout="wide")


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"No trained model at `{MODEL_PATH}`. Run `python -m olist_reviews.train` first.")
    st.stop()

artifact = load_artifact()
pipeline, defaults, options = artifact["pipeline"], artifact["defaults"]["values"], artifact["defaults"]["options"]
metrics = artifact["metrics"]

with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag the order when the risk is at least", 0.05, 0.80, float(round(artifact["threshold"], 2)), 0.01,
        help="The default maximises F1 on out-of-fold training predictions.")
    st.markdown("---")
    st.caption(
        f"Model: XGBoost pipeline · hold-out ROC-AUC {metrics['roc_auc']:.3f} · PR-AUC {metrics['pr_auc']:.3f}. "
        f"Among the 10% riskiest test orders, {metrics['precision_top10']:.0%} got a bad review "
        f"(base rate {metrics['base_rate']:.0%}).")

st.title("📦 Will this order get a bad review?")
st.write("Describe a delivered order. Fields you leave alone keep typical values from the training data.")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Basket")
    num_items = st.number_input("Items", 1, 20, 1)
    num_sellers = st.number_input("Sellers shipping it", 1, 5, 1, help="Each seller ships a separate parcel.")
    category = st.selectbox("Main product category", options["category"])
    total_price = st.number_input("Order value (R$)", 1.0, 5000.0, float(defaults["total_price"]), 10.0)
    total_freight = st.number_input("Freight (R$)", 0.0, 500.0, float(defaults["total_freight"]), 5.0)
with col2:
    st.subheader("Delivery")
    promised_days = st.slider("Promised delivery time (days)", 3, 60, round(defaults["promised_days"]))
    delivery_days = st.slider("Actual delivery time (days)", 1, 60, round(defaults["delivery_days"]))
    customer_state = st.selectbox("Customer state", options["customer_state"])
    seller_state = st.selectbox("Seller state", options["seller_state"])
with col3:
    st.subheader("Seller")
    seller_prior_reviews = st.number_input("Reviews the seller already has", 0, 2000,
                                           int(defaults["seller_prior_reviews"]))
    seller_prior_bad_rate = st.slider("Share of them that were 1-3 stars", 0.0, 1.0,
                                      float(round(defaults["seller_prior_bad_rate"], 2)), 0.01,
                                      disabled=seller_prior_reviews == 0)
    payment_type = st.selectbox("Payment", options["payment_type"])
    installments = st.number_input("Installments", 1, 24, int(defaults["installments"]))

delay = delivery_days - promised_days
row = {
    **defaults,
    "num_items": num_items, "num_products": num_items,
    "num_sellers": min(num_sellers, num_items), "category": category,
    "total_price": total_price, "total_freight": total_freight, "freight_ratio": total_freight / total_price,
    "promised_days": promised_days, "delivery_days": delivery_days, "delay_days": delay, "is_late": int(delay > 0),
    "customer_state": customer_state, "seller_state": seller_state, "same_state": int(customer_state == seller_state),
    "seller_prior_reviews": seller_prior_reviews,
    "seller_prior_bad_rate": seller_prior_bad_rate if seller_prior_reviews else float("nan"),
    "payment_type": payment_type, "installments": installments,
}
X = pd.DataFrame([row])[artifact["features"]]
risk = float(pipeline.predict_proba(X)[0, 1])

st.markdown("---")
left, right = st.columns([1, 2])
with left:
    vs_average = risk / metrics["base_rate"] - 1
    st.metric("Risk of a 1-3 star review", f"{risk:.0%}", f"{vs_average:+.0%} vs an average order",
              delta_color="inverse")
    st.progress(min(risk, 1.0))
    if risk >= threshold:
        st.error("**Flagged.** Reach out before the customer reviews: apology, voucher or a status update.")
    else:
        st.success("**Not flagged.** No intervention needed.")
    if delay > 0:
        st.caption(f"Delivered {delay} day(s) after the promised date.")
    elif delay < 0:
        st.caption(f"Delivered {-delay} day(s) ahead of the promised date.")

with right:
    st.subheader("What drives this prediction")
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X)
    contrib = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    drivers = pd.Series(contrib, index=[LABELS.get(c, c.replace("payment_type_", "payment: ")) for c in Xt.columns])
    top = drivers.reindex(drivers.abs().sort_values(ascending=False).index[:8]).rename_axis("feature")
    chart = alt.Chart(top.rename("contribution").reset_index()).mark_bar(cornerRadius=3).encode(
        x=alt.X("contribution:Q", title="push towards a bad review (log-odds)"),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.condition("datum.contribution > 0", alt.value("#dc2626"), alt.value("#16a34a")),
        tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
    )
    st.altair_chart(chart, width="stretch")
    st.caption("SHAP contributions from XGBoost: bars to the right raise the risk, bars to the left lower it.")
