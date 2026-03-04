from src.db import get_engine
import pandas as pd
from typing import Optional

def get_sql_data(limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Fetches data from the 'features_view' from the database.

    Args:
        limit (int, optional): Max number of rows to fetch. By default all of the rows are fetched.

    Returns:
        Optional[pd.DataFrame]: DataFrame with the query result or None in case of a connection or a data-fetching error.
    """
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