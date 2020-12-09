import logging
import tensorflow as tf
from collections import namedtuple
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
        self.model = namedtuple('Model', [
            'trend_disc_model',
            'trend_gen_model',
            'stationary_disc_model',
            'stationary_gen_model',
            'trend_disc_opt',
            'trend_gen_opt',
            'stationary_disc_opt',
            'stationary_gen_opt',
            'gen_loss',
            'disc_loss',
        ])

    def optimizer(self):
        # Set params as tf.Variable else the values are not tracked in ckpt
        # See https://github.com/tensorflow/tensorflow/issues/33150
        opt = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(self.config['training']['optimizer']['learning_rate']),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7))
        opt.iterations
        opt.decay = tf.Variable(0.0)
        return opt

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

        self.model = self.model(
            trend_disc_model=discriminator(),
            trend_gen_model=generator(),
            stationary_disc_model=discriminator(),
            stationary_gen_model=generator(),
            trend_disc_opt=self.optimizer(),
            trend_gen_opt=self.optimizer(),
            stationary_disc_opt=self.optimizer(),
            stationary_gen_opt=self.optimizer(),
            gen_loss=generator_loss,
            disc_loss=discriminator_loss,
        )    

        start_training(
            train_config=self.config['training'],
            ckpt_config=self.config['checkpoint'],
            trend_dataset=trend_dataset,
            stationary_dataset=stationary_dataset,
            model=self.model)
