"""Database connection. DATABASE_URL comes from the environment or a .env file."""
import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

DEFAULT_URL = "postgresql+psycopg://olist:olist@localhost:5432/olist"  # matches docker-compose.yml


def get_engine() -> Engine:
    load_dotenv()
    return create_engine(os.getenv("DATABASE_URL", DEFAULT_URL))
