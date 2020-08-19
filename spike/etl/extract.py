import os
import logging
import pandas as pd
import psycopg2 as pg
import numpy as np
from psycopg2 import sql
from dotenv import load_dotenv

PATH = './data/TEST'

DEFAULT_COLS = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
TRANSFORMED_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

LOGGER = logging.getLogger(__name__)

DOTENV_PATH = os.path.join(os.getcwd(), '.env')
load_dotenv(DOTENV_PATH)

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


def extract_csv(directory: str) -> pd.DataFrame:
    df = pd.DataFrame()
    LOGGER.info(f'Extracting csv from {directory} into dataframe')
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            ext = os.path.splitext(f)[1]
            if ext == '.csv':
                filepath = os.path.join(root, f)
                _df = pd.read_csv(filepath, header=None)
                df = df.append(_df, ignore_index=True)
    df.columns = DEFAULT_COLS
    # Combine Date and Time columns
    df['datetime'] = df['date'] + " " + df['time']
    df = df.drop(['date', 'time'], axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y.%m.%d %H:%M")
    df = df[TRANSFORMED_COLS]
    df = df.set_index('datetime')
    return df


def create_table(table: str) -> None:
    query = ('CREATE TABLE {}'
             '(datetime timestamp NOT NULL,'
             'open numeric NOT NULL,'
             'high numeric NOT NULL,'
             'low numeric NOT NULL,'
             'close numeric NOT NULL,'
             'volume numeric NOT NULL,'
             'CONSTRAINT %s_pkey PRIMARY KEY (datetime))') % (table)

    conn = None
    try:
        conn = pg.connect(**params)
        LOGGER.info(f'Creating new table {table} in postgres')
        cur = conn.cursor()
        sql_query = sql.SQL(query).format(sql.Identifier(table))
        LOGGER.debug(f'Executing postgres query = {sql_query.as_string(conn)}')
        cur.execute(sql_query)
        conn.commit()
        cur.close()
    except (Exception, pg.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()


def load_new_dataset(df: pd.DataFrame, table: str) -> None:
    """Fastest load for postgres by using the copy_from function.
    This will have errors if there is are duplicate rows.

    Args:
        df (pd.DataFrame): trading data
        table (str): postgres table name
    """
    tmp_df = '/tmp/tmp_dataframe.csv'
    df.to_csv(tmp_df, index_label='datetime', header=False)
    f = open(tmp_df, 'r')

    conn = None
    try:
        conn = pg.connect(**params)
        LOGGER.info(f'Copying to table {table} in postgres')
        cur = conn.cursor()
        cur.copy_from(f, table, sep=",")
        conn.commit()
        cur.close()
    except (Exception, pg.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()
        os.remove(tmp_df)


def upsert_dataset(df: pd.DataFrame, table: str) -> None:
    """This is a slower update and intended to allow duplicate rows.

    Args:
        df (pd.DataFrame): trading data
        table (str): postgres table name
    """
    data = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    query = ('INSERT INTO %s (%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s)'
             'ON CONFLICT DO NOTHING' % (table, cols))
    conn = None
    try:
        conn = pg.connect(**params)
        LOGGER.info(f'Upserting to table {table} in postgres')
        cur = conn.cursor()
        cur.executemany(query, data)
        conn.commit()
        cur.close()
    except (Exception, pg.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()


def read_postgres(table: str) -> pd.DataFrame:
    query = 'SELECT * from %s' % (table)
    conn = None
    df = None
    try:
        conn = pg.connect(**params)
        df = pd.read_sql_query(query, conn)
    except (Exception, pg.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()
        return df


def data_to_seq(data, timesteps=10):
    data_dim = data.shape[1]
    pad_rows = np.zeros((timesteps-1, timesteps, data.shape[1]))
    nrows = data.shape[0] - timesteps + 1
    p, q = data.shape
    m, n = data.strides
    strided = np.lib.stride_tricks.as_strided
    data = strided(data, shape=(nrows, timesteps, q), strides=(m, m, n))
    data = np.reshape(data, (-1, timesteps, data_dim))
    data = np.vstack((pad_rows, data))
    return(data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dirpath = './data/EURUSD'
    df = extract_csv(dirpath)
    table = 'eurusd_m1'
    create_table(table)
    load_new_dataset(df, table)
