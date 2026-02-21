from src.db import get_engine
import pandas as pd

#function for fetching data from the features_view view in the DB
def get_sql_data(limit:int = None):

    engine = get_engine()
    if engine is None:
        print('Unable to fetch the data from the DB!')     
        return None 
        
    query = 'SELECT * FROM features_view'
    if limit:
        query += f' LIMIT {limit}'

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print('Unable to fetch the data from the DB!')
        return None
    
    return df