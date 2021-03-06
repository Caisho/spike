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
        # data = self._add_price_changes(data)
        data = self._convert_to_sequence(data, seq_len=self.config['seq_len'])
        trend, stationary = self._split_trending_stationary(data)
        # trend = self._cumulative_sum(trend, 0)
        # stationary = self._cumulative_sum(stationary, 0)
        batch_trend = self._data_generator(trend)
        batch_stationary = self._data_generator(stationary)
        return batch_trend, batch_stationary

    @staticmethod
    def _add_price_changes(df: pd.DataFrame) -> pd.DataFrame:
        # vs present bar
        df['close-open'] = df['close'] - df['open']
        df['high-open'] = df['high'] - df['open']  # >= 0
        df['open-low'] = df['open'] - df['low']  # >= 0
        df['high-close'] = df['high'] - df['close']  # >= 0
        df['close-low'] = df['close'] - df['low']  # >= 0
        df['high-low'] = df['high'] - df['low']  # >= 0
        # same param vs prev bar
        df['high-high1'] = df['high'] - df['high'].shift()
        df['low-low1'] = df['low'] - df['low'].shift()
        df['open-open1'] = df['open'] - df['open'].shift()
        df['close-close1'] = df['close'] - df['close'].shift()
        # open vs prev bar close
        df['open-close1'] = df['open'] - df['close'].shift()
        # remove NaN row created because of shift
        df = df.iloc[1:]
        return df

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
    def _cumulative_sum(data, col_num):
        """ Concatenate the cumulative sum of one column in a 3-dim array to original array

        Args:
            data (np.ndarray): 3-dim array
            col_num (int): column num to be cumlative sum and concatenated

        Returns:
            [np.ndarray]: 3-dim array with additional cumulative sum column
        """
        cumsum = np.cumsum(data, axis=1)
        cumsum = cumsum[:, :, col_num].reshape((data.shape[0], data.shape[1], 1))
        return np.concatenate((data, cumsum), axis=2)

    @staticmethod
    def _split_trending_stationary(data: np.ndarray, spread: float = 0.0003) -> (np.ndarray, np.ndarray):
        """ This function takes in a 3d array of prices in shape (row, seq, features)
        For each row, we check whether the difference in the close price at the end and start of the sequence
        is greater than max of ATR and estimated bid-ask spread

        If greater -> we consider this sequence to be trending and append it to a trending list
        Else -> we consider this sequence to be stationary and append it to a stationary list

        Assume that close price is at data[:, :, 3] and atr[:, : 4]
        And price changes is >= [:, :, 5] 

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
                trending.append(data[i, :, :4])  # do not include price and atr columns
            else:
                stationary.append(data[i, :, :4])  # do not include price and atr columns
        return np.array(trending), np.array(stationary)

    def _data_generator(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.cache()
        dataset = dataset.shuffle(len(data), reshuffle_each_iteration=True)
        batch_dataset = dataset.batch(self.config['batch_size'])
        return batch_dataset

    @staticmethod
    def convert_to_price(seq, open_price=1):
        """ convert numpy sequence of price changes into sequence of price [open, high, low, price]

            col0: close-open
            col1: high-open
            col2: open-low
            col3: high-close
            col4: close-low
            col5: high-low
            col6: high-(prev high)
            col7: low-(prev low)
            col8: open-(prev open)
            col9: close-(prev close)
            col10: open-(prev close)

        Args:
            seq (np.ndarray): shape (seq_len, 11)

        Returns:
            [np.ndarray]: shape (seq_len, 4)
        """
        prices = []
        high_price = 0
        low_price = 0
        close_price = 0
        for row in seq:
            close_price = row[0] + open_price
            high_price = row[1] + open_price
            low_price = -(row[2] - open_price)
            prices.append((open_price, high_price, low_price, close_price))
            open_price = close_price
        return np.array(prices)
