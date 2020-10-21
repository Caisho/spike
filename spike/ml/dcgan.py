import os
import time
import tensorflow as tf
from random import random, randint
from tensorflow.keras import layers
from spike.etl.extract import read_postgres
from spike.etl.transform import atr, to_sequence, split_trending_stationary
from mlflow import log_metric, log_param, log_artifacts


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4096, input_shape=(32,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((128, 32)))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(16, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(8, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1D(4, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 3, strides=2, input_shape=[512, 4], padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 3, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# df = read_postgres('eurusd_m1', start_date='2018-01-01')
# df = atr(df)
# data = to_sequence(df, 50)
# trend, stationary = split_trending_stationary(data, spread=0.0003)

# trend_generator = make_generator_model()
# stationary_generator = make_generator_model()

# trend_discriminator = make_discriminator_model()
# stationary_discriminator = make_discriminator_model()

# noise = tf.random.normal([1,32])

# test = trend_generator(noise, training=False)

# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  trend_generator=trend_generator,
#                                  stationary_generator=stationary_generator,
#                                  trend_discriminator=trend_discriminator,
#                                  stationary_discriminator=stationary_discriminator)

# BATCH_SIZE = 10
# EPOCHS = 50
# noise_dim = 100
# num_examples_to_generate = 16

# # We will reuse this seed overtime (so it's easier)
# # to visualize progress in the animated GIF)
# seed = tf.random.normal([num_examples_to_generate, noise_dim])


# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_trend = trend_generator(noise, training=True)
#         real_trend = trend_discriminator(images, training=True)
#         fake_trend = trend_discriminator(generated_trend, training=True)
#         trend_gen_loss = generator_loss(fake_trend)
#         trend_disc_loss = discriminator_loss(real_trend, fake_trend)

#         gradients_of_trend_generator = gen_tape.gradient(trend_gen_loss, trend_generator.trainable_variables)
#         gradients_of_trend_discriminator = disc_tape.gradient(trend_disc_loss, trend_discriminator.trainable_variables)

#         generator_optimizer.apply_gradients(zip(gradients_of_trend_generator, trend_generator.trainable_variables))
#         discriminator_optimizer.apply_gradients(zip(gradients_of_trend_discriminator, trend_discriminator.trainable_variables))


if __name__ == "__main__":
    log_param('noise', randint(0, 100))
    log_metric('foo', random())
