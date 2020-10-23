import logging
import numpy as np
import pandas as pd
from spike.etl.extract import read_postgres


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


def split_trending_stationary(data: np.ndarray, spread: float = 0.0003) -> (np.ndarray, np.ndarray):
    """ This function takes in a 3d array of prices in shape (row, seq, features)
    For each row, we check whether the difference in the close price at the end and start of the sequence
    is greater than max of ATR and estimated bid-ask spread

    If greater -> we consider this sequence to be trending and append it to a trending list
    Else -> we consider this sequence to be stationary and append it to a stationary list

    Assume that close price is at data[:, :, 3] and atr[:, : 4]

    Args:
        data (np.ndarray): numpy array
        spread (float): estimated bid-ask spread in points

    Returns:
        (np.ndarray, np.ndarray): tuple of numpy array in shape of (row, seq, features)
    """
    trending = []
    stationary = []

    for i in range(len(data)):
        close = data[i, :, 3]
        price_change = abs(close[-1] - close[0])
        price_atr = data[i, 0, 4]
        if price_change > max(spread, price_atr):
            trending.append(data[i, :, 0:4])  # remove atr column
        else:
            stationary.append(data[i, :, 0:4])  # remove atr column
    return (np.array(trending), np.array(stationary))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    table = 'eurusd_m1'
    df = read_postgres(table, start_date='2018-06-01')
    df = atr(df)
    df = df.drop(['volume'], axis=1)
    data = to_sequence(df, 50)
    trend, stationary = split_trending_stationary(data)
