from db import get_engine
import pandas as pd

def get_sql_data(limit:int = None):

    engine = get_engine()
    query = 'SELECT * FROM features_view'

    if limit:
        query += f' LIMIT {limit}'

    return pd.read_sql(query, engine)