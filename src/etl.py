import pandas as pd
from typing import Dict

#Function extracting raw data from a CSV file
def extract_csv(file_path: str, dtype_dict: Dict =None):
    print(f'Extracting data from {file_path}')

    return pd.read_csv(file_path, dtype=dtype_dict)

#Each table has its unique transform function, based on the data types requirements in the DB
def transform_customers(df: pd.DataFrame):
    df['customer_city'] = df['customer_city'].str.lower()
    return df

def transform_products(df: pd.DataFrame):
    df['product_category_name'] = df['product_category_name'].fillna('unknown')
    #df['product_photos_qty'] = df['product_photos_qty'].fillna(0)

    return df
    
def transform_orders(df: pd.DataFrame):
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format="%Y-%m-%d %H:%M:%S")

    return df

def transform_reviews(df: pd.DataFrame):
    text_cols = ['review_comment_title', 'review_comment_message']
    date_cols = ['review_creation_date', 'review_answer_timestamp']

    for col in text_cols:
        df[col] = df[col].fillna('')
        df[col] = df[col].str.replace(r'[\n\r]',' ', regex=True)
        df[col] = df[col].str.strip()

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format="%Y-%m-%d %H:%M:%S")
    
    return df

def transform_order_items(df: pd.DataFrame):
    df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")

    return df

#Function responsible for loading transformed data into the DB (due to the size of the data it's performed in chunks of 1000)   
def load_db(df: pd.DataFrame, con_engine, table_name: str):
    try:
        df.to_sql(
            name=table_name,
            con=con_engine,
            index=False,
            if_exists='append',
            chunksize=1000,
            method='multi'
        )
        print(f'Succesfully loaded data to the {table_name} table')
    except Exception as e:
        print(f'Error while loading to the {table_name} table: {e}')

