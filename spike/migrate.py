import os
import logging
import psycopg2 as pg
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')

params = {
    'host': POSTGRES_HOST,
    'port': POSTGRES_PORT,
    'user': POSTGRES_USER,
    'password': POSTGRES_PASSWORD,
    'dbname': POSTGRES_DB
}


def create_eurjpy_table():
    query = (
        'CREATE TABLE public.eurusd_m1 ('
        '   datetime timestamp NOT NULL,'
        '   open numeric NOT NULL,'
        '   high numeric NOT NULL,'
        '   low numeric NOT NULL,'
        '   close numeric NOT NULL,'
        '   volume numeric NOT NULL,'
        '   CONSTRAINT eurusd_m1_pkey PRIMARY KEY (datetime)'
        ');'
    )
    conn = None
    try:
        conn = pg.connect(**params)
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
    except pg.Error as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)

    create_eurjpy_table()
