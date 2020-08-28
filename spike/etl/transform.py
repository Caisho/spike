import logging
import numpy as np
import pandas as pd
from typing import List
from spike.etl.extract import read_postgres

SPREAD = 0.0003


def atr(df: pd.DataFrame) -> pd.DataFrame:
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/100, adjust=False).mean()
    df = df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
    return df


def to_sequence(df: pd.DataFrame, seq_len: int = 100) -> np.ndarray:
    data = df.to_numpy()
    _len = len(data) - (len(data) % seq_len)
    data = data[:_len, :]
    data = data.reshape(-1, seq_len, data.shape[-1])
    return data


def split_trending_stationary(data: np.ndarray) -> (List, List):
    trending = []
    stationary = []

    for i in range(len(data)):
        close = data[i, :, 4]
        price_change = abs(close[-1] - close[0])
        price_atr = data[i, 0, 5]
        if price_change > max(SPREAD, price_atr):
            trending.append(data[i])
        else:
            stationary.append(data[i])
    return (trending, stationary)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    table = 'eurusd_m1'
    df = read_postgres(table, start_date='2018-06-01')
    df = atr(df)
    df = df.drop(['volume'], axis=1)
    data = to_sequence(df, 50)
    trend, stationary = split_trending_stationary(data)
