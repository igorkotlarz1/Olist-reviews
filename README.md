# Olist E-Commerce: Predicting Customer Dissatisfaction  
> The main goal of this project was to analyze and explore the **Olist** (Brazilian e-commerce platform) dataset to build an **End-to-End Machine Learning pipeline**. This system predicts  customer dissatisfaction (the risk of receiving a 1-3 star review) based on logistical metadata.
---
## By analyzing logistical data (delivery timestamps, item counts, shipping costs) after the delivery phase, our model identified key factors driving negative reviews:
* **Multi-item orders** drastically increase the risk of a bad review. The most probable explanation is logistical complexity of split-shipments such as delayed packaging or missing parts.
* **Late orders** (where the actual delivery time exceeds the platform's estimated date) act as dealbreakers and have strong effect on the final dissatisfaction.
* **Long delivery times** have an impact on the final review, yet not that significant as the **unexpected lateness**. 

## Pipeline Architecture
1. **ETL (Extract, Transform, Load) -** Raw CSV data from the [Kaggle repository](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) was extracted, processed (initially cleaned and transformed) and loaded into a local **PostgreSQL** database using **SQLAlchemy** library. A dedicated view `features_view` was created in SQL to aggregate the data and prepare it for exploration and modelling.
2. **Feature Engineering and Preprocessing -** While basic feature engineering was performed directly in SQL, a custom `Transformer` class handles more complex logic: **log transformation** of highly-skewed features, **outlier clipping** and **encoding** categorical feature using a `TargetEncoder`.
3. **Modelling and Tuning -** Custom `Selector` class handles feature selection using Recursive Feature Elimination (RFECV). The main `Model` class is responsible for hyperparameter tuning via **Optuna** and cross-validation training.
4. **Deployment -** A fully interactive **Streamlit** web dashboard was built to help analyze the key factors responsible for customer dissatisfaction by tweaking specific order parameters.

## Tech Stack
* **Database:** PostgreSQL, SQLAlchemy
* **Data Processing:** Python, Pandas, NumPy, Category Encoders
* **Machine Learning and Hyperparameter Tuning:** XGBoost, Scikit-Learn, Optuna
* **Visualizations:** Matplotlib, Seaborn, SHAP
* **Deployment:** Streamlit
