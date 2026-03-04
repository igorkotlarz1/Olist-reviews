from src.db import get_engine
import src.etl as etl
import os

from typing import Any, Dict

PIPELINE_CONFIG: Dict[str, Dict[str, Any]] = {
    'customers' : 
    {
        'path': os.path.join('data', 'olist_customers_dataset.csv'),
        'func': etl.transform_customers,
        'dtypes': {'customer_zip_code_prefix': str}
    },

    'products' : 
    {
        'path': os.path.join('data', 'olist_products_dataset.csv'),
        'func': etl.transform_products,
        'dtypes': {
            'product_name_lenght': 'Int64',
            'product_description_lenght': 'Int64',
            'product_photos_qty': 'Int64',
            'product_weight_g': 'Int64',
            'product_length_cm': 'Int64',
            'product_height_cm': 'Int64',
            'product_width_cm': 'Int64'}
    },

    'orders' : 
    {
        'path': os.path.join('data','olist_orders_dataset.csv'),
        'func': etl.transform_orders,
        'dtypes': None
    },

    'reviews' :
    {
        'path': os.path.join('data','olist_order_reviews_dataset.csv'),
        'func': etl.transform_reviews,
        'dtypes': None
    },

    'order_items' :
    {
        'path': os.path.join('data','olist_order_items_dataset.csv'),
        'func': etl.transform_order_items,
        'dtypes': None
    },

    'category_names' :
    {
        'path': os.path.join('data','product_category_name_translation.csv'),
        'func': etl.transform_category_names,
        'dtypes': None
    }
}

def main() -> None:
    """
    The main entry point for the ETL pipeline execution.
    
    Orchestrates the connection to the database, iterates over the defined PIPELINE_CONFIG, extracts data from CSV files, applies  transformations, and loads the data into the corresponding tables.
    """
    try:
        engine = get_engine()

        if engine is None:
            print("Failed to create the engine")
            return
        
    except Exception as e:
        return
    
    #running ETL pipeline
    with engine.begin() as conn: 
        for table_name, config in PIPELINE_CONFIG.items():
            path = config['path']
            dtypes = config['dtypes']
            
            try:
                #extracting CSV data into a DataFrame
                df = etl.extract_csv(path, dtypes)

                transform_func = config['func']

                #transforming the data
                df_transformed = transform_func(df)

                #loading the data into the corresponding table in the postgres DB
                etl.load_db(df_transformed, conn, table_name)

            except Exception as e:
                print(f'An error occured while loading table {table_name}: {e}')
                break       

if __name__ == "__main__":
    main()

