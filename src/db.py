from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from typing import Optional, Any

load_dotenv()

def get_db_url():
    """
    Retrieves the database URL from the enviroment variables.

    Raises:
        ValueError: If DATABASE_URL is not defined in the .env file.

    Returns:
        str: Database URL connection address.
    """
    url = os.getenv("DATABASE_URL")

    if not url:
        raise ValueError('DATABASE_URL is not definied in the .env file!')
    return url

def get_engine() -> Optional[Any]:
    """
    Creates and returns the database engine (SQLAlchemy Engine).

    Returns:
        Optional[Any]: SQLAlchemy engine object or None in case of connection failure.
    """
    url = get_db_url()

    try:
        engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        print(f'Error while creating DB engine {e}')
        return None