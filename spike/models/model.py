import os
import logging
import tensorflow as tf
import altair as alt
import numpy as np
import pandas as pd
import wandb
from collections import namedtuple
from train.load_config import load_configs
from models.layers import generator, discriminator
from models.loss import generator_loss, discriminator_loss
from dataset.dataset import FxDataset
from train.trainer.functions import start_training, restore_checkpoint

LOGGER = logging.getLogger(__name__)


class DcganModel():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()
        self.model = namedtuple('Model', [
            'trend_disc_model',
            'trend_gen_model',
            'stat_disc_model',
            'stat_gen_model',
            'trend_disc_opt',
            'trend_gen_opt',
            'stat_disc_opt',
            'stat_gen_opt',
            'gen_loss',
            'disc_loss',
        ])
        self.model = self.model(
            trend_disc_model=discriminator(self.config['training']['input_dim'], self.config['training']['seq_len']),
            trend_gen_model=generator(self.config['training']['input_dim'], self.config['training']['seq_len'], self.config['training']['noise_dim']),
            stat_disc_model=discriminator(self.config['training']['seq_len'], self.config['training']['input_dim']),
            stat_gen_model=generator(self.config['training']['input_dim'], self.config['training']['seq_len'], self.config['training']['noise_dim']),
            trend_disc_opt=self.optimizer(),
            trend_gen_opt=self.optimizer(),
            stat_disc_opt=self.optimizer(),
            stat_gen_opt=self.optimizer(),
            gen_loss=generator_loss,
            disc_loss=discriminator_loss,
        )

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

    def generate_and_save(self, epoch):
        self.model = restore_checkpoint(ckpt_config=self.config['checkpoint'], model=self.model)
        noise = tf.random.normal([self.config['training']['batch_size'], self.config['training']['noise_dim']])
        gen_data = self.model.trend_gen_model(noise)[np.random.randint(64)]

        # np.savetxt(os.path.join(wandb.run.dir, 'epoch-'+epoch+'-features.csv'), gen_data, delimiter=',')
        price = FxDataset().convert_to_price(gen_data)
        # np.savetxt(os.path.join(wandb.run.dir,  'epoch-'+epoch+'-prices.csv'), gen_data, delimiter=',')
        LOGGER.info('Generated sample features and prices')

        price = pd.DataFrame(price, columns=['Open', 'High', 'Low', 'Close']).reset_index()
        chart = alt.Chart(price).mark_area(
            color="lightblue",
            interpolate='step-after',
            line=True
        ).encode(
            alt.Y('Close:Q', scale=alt.Scale(zero=False)),
            alt.X('index:Q', axis=alt.Axis(title='Timestep'))
        )
        chart.save(os.path.join(wandb.run.dir,  'epoch-'+epoch+'-chart.html'))

    def train(self):
        tf.random.set_seed(1234)
        trend_dataset, stationary_dataset = FxDataset().get_dataset()

        start_training(
            train_config=self.config['training'],
            ckpt_config=self.config['checkpoint'],
            trend_dataset=trend_dataset,
            stat_dataset=stationary_dataset,
            model=self.model)
