import tensorflow as tf
import logging
from models.model import DcganModel

import os
from train.load_config import load_configs

# prevent gpu running out of memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    dcgan_model = DcganModel()
    dcgan_model.train()
