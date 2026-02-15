from dotenv import load_dotenv
from sqlalchemy import create_engine
import os


load_dotenv()

def get_db_url():
    #loading DB connection url from the .env file
    url = os.getenv("DATABASE_URL")

    if not url:
        raise ValueError('DATABASE_URL is not definied in the .env file!')
    return url

def get_engine():
    url = get_db_url()

    try:
        engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        print(f'Error while creating DB engine {e}')
        return None