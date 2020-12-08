import logging
import tensorflow as tf
from train.load_config import load_configs
from models.layers import generator, discriminator
from models.loss import generator_loss, discriminator_loss
from dataset.dataset import FxDataset
from train.trainer.functions import start_training

LOGGER = logging.getLogger(__name__)


class DcganModel():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()

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

    def train(self):
        trend_dataset, stationary_dataset = FxDataset().get_dataset()
        trend_generator, trend_discriminator = self.create_models()

        # Set params as tf.Variable else the values are not tracked in ckpt
        # See https://github.com/tensorflow/tensorflow/issues/33150
        generator_optimizer = discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(self.config['training']['optimizer']['learning_rate']),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7))
        generator_optimizer.iterations
        generator_optimizer.decay = tf.Variable(0.0)

        start_training(
            train_config=self.config['training'],
            ckpt_config=self.config['checkpoint'],
            trend_dataset=trend_dataset,
            stationary_dataset=stationary_dataset,
            generator_model=trend_generator,
            discriminator_model=trend_discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss)
