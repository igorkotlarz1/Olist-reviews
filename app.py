import streamlit as st

import pandas as pd
import numpy as np
import joblib

categories = ["agro_industry_and_commerce","air_conditioning","art","arts_and_craftmanship","audio","auto","baby","bed_bath_table","books_general_interest","books_imported","books_technical","cds_dvds_musicals","christmas_supplies","cine_photo","computers","computers_accessories","consoles_games","construction_tools_construction","construction_tools_lights","construction_tools_safety","cool_stuff","costruction_tools_garden","costruction_tools_tools","diapers_and_hygiene","drinks","dvds_blu_ray","electronics","fashio_female_clothing","fashion_bags_accessories","fashion_childrens_clothes","fashion_male_clothing","fashion_shoes","fashion_sport","fashion_underwear_beach","fixed_telephony","flowers","food","food_drink","furniture_bedroom","furniture_decor","furniture_living_room","furniture_mattress_and_upholstery","garden_tools","health_beauty","home_appliances","home_appliances_2","home_comfort_2","home_confort","home_construction","housewares","industry_commerce_and_business","kitchen_dining_laundry_garden_furniture","la_cuisine","luggage_accessories","market_place","music","musical_instruments","office_furniture","party_supplies","perfumery","pet_shop","security_and_services","signaling_and_security","small_appliances","small_appliances_home_oven_and_coffee","sports_leisure","stationery","tablets_printing_image","telephony","toys","watches_gifts","unknown"]

st.set_page_config(page_title='Olist Customer Satisfaction', page_icon='📦',layout='wide')

@st.cache_resource
def load_assets():
    xgb = joblib.load('models/best_xgb.pkl')
    rf = joblib.load('models/best_rf.pkl')
    encoder = joblib.load('models/target_encoder.pkl')

    return xgb, rf, encoder

try:
    xgb, rf, encoder = load_assets()
except Exception as e:
    st.error("Error occured while loading the models! Make sure appropriate files are in the /models directory!")
    st.stop()

with st.sidebar:
    st.header("Model selection")
    model_selection = st.selectbox("Select the prediction model", ('XGBoost', 'Random Forest'))
    st.markdown("---")

    st.header("Threshold selection")
    threshold = st.slider("Select the prediction probability threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.01,
                          help="The customer will be considered dissatisfied if the predicted probability (of bad review score) is above this threshold")

st.title("Customer satisfaction prediction dashboard")
st.markdown("---")
st.header("Enter the order parameters to estimate the risk of customer leaving a negative review score (1-3 stars).")

col1, col2 = st.columns(2)

# Setting order parameters 
with col1:
    st.subheader("Order details")
    num_items = st.number_input("Number of items in the order", min_value=1, value=1, step=1)
    desc_len = st.number_input("Most important product's description length", min_value=0, value=750, step=50)
    category = st.selectbox("Most important product's category", categories)
    volume_l = st.number_input("Most important product's volume (liters)", min_value=0.0, value=10.0, step=5.0)

with col2:
    st.subheader("Delivery details")
    delivery_days = st.slider("Delivery time (days)", min_value=1, max_value=50, value=5, step=1)
    estimated_delivery_diff = st.slider("Difference between the actual delivery and the estimated delivery date (days)", min_value=-30, max_value=30, value=-5,step=1)
    freight_value = st.number_input("Total freight value (R$)", min_value=0.0, value=20.0, step=5.0, help="For all items in the order")

st.markdown("---")

# Making predictions
if st.button("Analyse the order", use_container_width=True, type="primary"):
    log_total_freight = np.log1p(freight_value)

    input_data = pd.DataFrame({
        'num_items':[num_items], 
        'desc_len': [desc_len], 
        'category': [category], 
        'volume_l': [volume_l], 
        'delivery_days': [delivery_days], 
        'estimated_delivery_diff': [estimated_delivery_diff], 
        'log_total_freight': [log_total_freight]   
    })

    input_data['category'] = encoder.transform(input_data[['category']])

    if model_selection == 'XGBoost':
        model = xgb
    elif model_selection == 'Random Forest':
        model = rf

    y_proba = model.predict_proba(input_data)[0, 1]

    st.markdown("---")
    st.subheader("Analysis Results")

    # Analysis results panel
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Risk of Negative Review", 
            value=f"{y_proba * 100:.1f}%",
            delta="High Risk" if y_proba >= threshold else "- Safe",
            delta_color="inverse")

    with res_col2:
        if y_proba >= threshold:
            st.error("**High Risk of Dissatisfaction!** The customer is likely to leave a 1-3 star review.")
            st.caption("Action required: Contact by the Customer Support team is recommended.")
        else:
            st.success("**Low Risk.** The customer is likely to be satisfied (4-5 stars).")
            st.caption("Action required: No intervention needed.")

    st.progress(float(y_proba), text="Current Risk Level")

    with st.expander("View Insights and Risk Factors"):
        st.write("Based on historical data model flagged the following factors:")
        if num_items >= 2:
            st.warning("**Multiple items:** Orders with 2+ items tend to double the risk of negative reviews.")

        if estimated_delivery_diff > 0:
            st.warning(f"**Delay:** The package is delayed by {estimated_delivery_diff} day(s) causing customer dissatisfaction.")
