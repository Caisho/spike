import logging
from train.load_config import load_configs
from models.layers import generator, discriminator
from dataset.dataset import FxDataset
from train.trainer.functions import train_loop
import wandb


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

    # TODO understand why tf.function has error
    # @tf.function
    def train(self):
        self.logger.info('Logging training config params to wandb')
        wandb.init(config=self.config, sync_tensorboard=True)
        # create dataset
        trend_dataset, stationary_dataset = FxDataset().get_dataset()
        # create models
        trend_generator, trend_discriminator = self.create_models()
        # run training
        train_loop(
            train_config=self.config['training'],
            ckpt_config=self.config['checkpoint'],
            dataset=trend_dataset,
            generator_model=trend_generator,
            discriminator_model=trend_discriminator)
