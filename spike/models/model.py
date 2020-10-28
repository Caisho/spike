import logging
import tensorflow as tf
from train.load_config import load_configs
from models.layers import generator, discriminator
from dataset.dataset import FxDataset
from train.trainer.functions import train_loop
from mlflow import log_params


class DcganModel():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()
        self.model_name = 'dcgan'
        self.logger.info('Dcgan configs summary')
        self.logger.info(self.config)

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

    # @tf.function
    def train(self):
        # log params to mlflow
        log_params(self.config['training'])
        # create dataset
        trend_dataset, stationary_dataset = FxDataset().get_dataset()
        # create models
        trend_generator, trend_discriminator = self.create_models()
        # run training
        train_loop(self.config['training'], trend_dataset, trend_generator, trend_discriminator)
