import logging
import os
import time
import tensorflow as tf
from train.load_config import load_configs

TRAIN_CONFIG = load_configs()['training']
CHECKPOINT_CONFIG = load_configs()['checkpoint']


def start_distributed_training():
    pass


def _get_checkpoint_path(checkpoint_dir, model_name, checkpoint_suffix):
    pass


def restore_checkpoint(model_name, model, optimizer, checkpoint_dir, checkpoint_suffix):
    pass


@tf.function
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


def train_loop(train_config, ckpt_config, dataset, num_epochs, validation_freq):
    for epoch in range(num_epochs):

        for batch_data in dataset:
            train_step(batch_data)

