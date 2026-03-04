import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset by removing NA values, duplicates and fixing invalid values.

    Args:
        df (pd.DataFrame): A DataFrame with raw data fetched from the database.

    Returns:
        pd.DataFrame: The cleaned DataFrame ready for feature engineering.
    """

    df_clean = df.dropna(axis=0) \
        .drop_duplicates(subset=['order_id'], keep='last').copy()
    
    cols_to_change = ['delivery_days', 'estimated_delivery_diff', 'seller_disp_diff', 'processing_days']

    for col in cols_to_change:
        df_clean[col] = df_clean[col].astype(int)

    df_clean.loc[df_clean['processing_days'] < 0 ,'processing_days'] = 0 

    df_clean.drop(['order_id'], axis=1, inplace=True)

    return df_clean