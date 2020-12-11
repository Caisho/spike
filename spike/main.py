import os
import logging
import tensorflow as tf
import wandb
from dotenv import load_dotenv
from models.model import DcganModel

LOGGER = logging.getLogger(__name__)

load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# prevent gpu running out of memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    dcgan_model = DcganModel()
    dcgan_model.train()
