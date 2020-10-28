import logging
import os
import time
import tensorflow as tf
from train.load_config import load_configs
from models.loss import generator_loss, discriminator_loss
from models.layers import generator, discriminator
from dataset.dataset import FxDataset

TRAIN_CONFIG = load_configs()['training']
CHECKPOINT_CONFIG = load_configs()['checkpoint']


def restore_checkpoint(model_name, model, optimizer, checkpoint_dir, checkpoint_suffix):
    pass


#@tf.function
def train_step(
        train_config,
        batch_data,
        generator_model,
        discriminator_model,
        generator_loss,
        discriminator_loss,
        generator_optimizer,
        discriminator_optimizer):

    noise = tf.random.normal([train_config['batch_size'], train_config['noise_dim']])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator_model(noise, training=True)

        real_output = discriminator_model(batch_data, training=True)
        fake_output = discriminator_model(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_variables))
    return gen_loss, disc_loss


def _train_one_epoch(train_config, dataset):
    logger = logging.getLogger(__name__)

    generator_model = generator()
    discriminator_model = discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    for batch_data in dataset:
        gen_loss, disc_loss = train_step(
            train_config=train_config,
            batch_data=batch_data,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer)
    logger.info(f'Generator loss = {gen_loss}')
    logger.info(f'Discriminator loss = {disc_loss}')


def train_loop(train_config):
    logger = logging.getLogger(__name__)
    dataset = FxDataset()
    trend_dataset, stationary_dataset = dataset.get_dataset()

    # TODO tensorboard summary, tensorboard logs, checkpoint, mlflow
    for epoch in range(train_config['num_epochs']):
        logger.info(f'Running epoch {epoch}')
        _train_one_epoch(train_config, trend_dataset)
