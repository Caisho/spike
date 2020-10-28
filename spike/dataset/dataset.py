import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from train.load_config import load_configs
from dataset.postgres import read_postgres


class FxDataset:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()['training']

    def get_dataset(self):
        self.logger.info('Reading from Postgres table')
        data = read_postgres(
            table=self.config['symbol'],
            start_date=self.config['start_date']
        )
        self.logger.info(f"Dataset: {self.config['symbol']} {self.config['start_date']} {data.shape}")
        data = self._atr(data)
        data = self._convert_to_sequence(data, seq_len=self.config['sequence_len'])
        trend, stationary = self._split_trending_stationary(data)
        batch_trend = self._data_generator(trend)
        batch_stationary = self._data_generator(stationary)
        return batch_trend, batch_stationary

    @staticmethod
    def _atr(df: pd.DataFrame) -> pd.DataFrame:
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].ewm(alpha=1/100, adjust=False).mean()
        df = df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
        return df

    @staticmethod
    def _convert_to_sequence(df: pd.DataFrame, seq_len: int = 100) -> np.ndarray:
        data = df.to_numpy()
        _len = len(data) - (len(data) % seq_len)
        data = data[:_len, :]
        data = data.reshape(-1, seq_len, data.shape[-1])
        return data

    @staticmethod
    def _split_trending_stationary(data: np.ndarray, spread: float = 0.0003) -> (np.ndarray, np.ndarray):
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
        return trending, stationary

    def _data_generator(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.cache()
        dataset = dataset.shuffle(len(data), reshuffle_each_iteration=True)
        batch_dataset = dataset.batch(self.config['batch_size'])
        return batch_dataset
