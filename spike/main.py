import tensorflow as tf
import logging
from mlflow import log_metric, log_param, log_artifacts
from train.trainer.functions import train_loop
from train.load_config import load_configs


TRAIN_CONFIG = load_configs()['training']
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    train_loop(TRAIN_CONFIG)
