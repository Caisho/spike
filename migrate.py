import os
import logging
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

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

schemas = ['sgh', 'nuslib']


def create_schema(schemas):
    query = 'CREATE SCHEMA {}'
    conn = None
    try:
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for schema in schemas:
            sql_query = sql.SQL(query).format(sql.Identifier(schema))
            LOGGER.debug(f'Executing postgres query = \
                {sql_query.as_string(conn)}')
            cur.execute(sql_query)
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()


def create_mapping_tables(schemas):
    query1 = 'CREATE TABLE {}.map_dataset_annotation \
    (dataset_table character varying(64) \
        COLLATE pg_catalog."default" NOT NULL, \
    annotation_table character varying(64) COLLATE \
        pg_catalog."default" NOT NULL, \
    CONSTRAINT dataset_annotation_pkey \
        PRIMARY KEY (dataset_table, annotation_table))'

    query2 = 'CREATE TABLE {}.map_dataset_split \
    (dataset_table character varying(64) \
        COLLATE pg_catalog."default" NOT NULL, \
    split_table character varying(64) COLLATE \
        pg_catalog."default" NOT NULL, \
    CONSTRAINT dataset_split_pkey PRIMARY KEY (dataset_table, split_table))'

    conn = None
    try:
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for schema in schemas:
            sql_query1 = sql.SQL(query1).format(sql.Identifier(schema))
            LOGGER.debug(f'Executing postgres query1 = \
                {sql_query1.as_string(conn)}')
            cur.execute(sql_query1)

            sql_query2 = sql.SQL(query2).format(sql.Identifier(schema))
            LOGGER.debug(f'Executing postgres query2 = \
                {sql_query2.as_string(conn)}')
            cur.execute(sql_query2)
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as e:
        LOGGER.exception(e)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    create_schema(schemas)
    create_mapping_tables(schemas)
