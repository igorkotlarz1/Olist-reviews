import pandas as pd
from typing import Dict, Optional, Any

#Function extracting raw data from a CSV file
def extract_csv(file_path: str, dtype_dict: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
    """
    Extracts raw data from a CSV file.

    Args:
        file_path (str): Filepath to a CSV file.
        dtype_dict (Optional[Dict[str, Any]], optional): Dict mapping column names to proper data types. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the extracted CSV data.
    """
    print(f'Extracting data from {file_path}')

    return pd.read_csv(file_path, dtype=dtype_dict)

#Each table has its unique transform function, based on the data types requirements in the DataBase

def transform_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the customers dataset by standardizing text columns.

    Args:
        df (pd.DataFrame): The raw customers DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame with lowercase customer cities.
    """
    df['customer_city'] = df['customer_city'].str.lower()
    return df

def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the products dataset by handling missing category values.

    Args:
        df (pd.DataFrame): The raw products DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame with missing category names replaced by 'unknown'.
    """
    df['product_category_name'] = df['product_category_name'].fillna('unknown')

    return df
    
def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the orders dataset by parsing datetime columns.

    Args:
        df (pd.DataFrame): The raw orders DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame with properly formatted datetime columns.
    """
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

def transform_reviews(df: pd.DataFrame)-> pd.DataFrame:
    """
    Transforms the order reviews dataset by cleaning text formats and parsing dates.

    Args:
        df (pd.DataFrame): The raw reviews DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame with cleaned text (removed newlines) and properly formatted datetime columns.
    """
    text_cols = ['review_comment_title', 'review_comment_message']
    date_cols = ['review_creation_date', 'review_answer_timestamp']

    for col in text_cols:
        df[col] = df[col].fillna('')
        df[col] = df[col].str.replace(r'[\n\r]',' ', regex=True)
        df[col] = df[col].str.strip()

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format="%Y-%m-%d %H:%M:%S")
    
    return df

def transform_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the order items dataset by parsing the shipping limit date.

    Args:
        df (pd.DataFrame): The raw order items DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame with the shipping_limit_date properly formatted.
    """
    df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")

    return df

def transform_category_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    A pass-through function to be used in the ETL pipeline.

    Args:
        df (pd.DataFrame): The raw category names DataFrame.

    Returns:
        pd.DataFrame: The unchanged DataFrame.
    """
    return df
 
def load_db(df: pd.DataFrame, con_engine:Any, table_name: str) -> None:
    """
    Loads transformed data into the PostgreSQL database, splitting the data into chunks of 1000 due to the size of the data.

    Args:
        df (pd.DataFrame): DataFrame to load.
        con_engine (Any): SQLAlchemy db engine.
        table_name (str): Name of the target table in the database.
    """
    try:
        df.to_sql(
            name=table_name,
            con=con_engine,
            index=False,
            if_exists='append',
            chunksize=1000,
            method='multi'
        )
        print(f'Succesfully loaded data into the {table_name} table')
    except Exception as e:
        print(f'Error while loading to the {table_name} table: {e}')

