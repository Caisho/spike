import logging
import tensorflow as tf
from train.load_config import load_configs
from models.layers import generator, discriminator
from dataset.dataset import FxDataset


class DcganModel():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()
        self.model_name = 'dcgan'

    def create_models(self):
        self.logger.info('Creating dcgan models')
        trend_generator = generator()
        trend_discriminator = discriminator()
        self.logger.info('Generator model summary')
        self.logger.info(trend_generator.summary())
        self.logger.info('Discriminator model summary')
        self.logger.info(trend_discriminator.summary())
        return trend_generator, trend_discriminator

    def predict(self, data):
        pass

    @tf.function
    def train(self, input_dir, output_dir, use_case, epochs=1):
        pass
