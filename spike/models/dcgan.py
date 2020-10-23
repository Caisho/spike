import os
import time
import tensorflow as tf
from random import random, randint
from tensorflow.keras import layers
from spike.etl.extract import read_postgres
from spike.etl.transform import atr, to_sequence, split_trending_stationary
from mlflow import log_metric, log_param, log_artifacts



if __name__ == "__main__":
    BATCH_SIZE = 10
    EPOCHS = 50
    SEQUENCE_LEN = 50
    SPREAD = 0.0003
    SYMBOL = 'eurusd_m1'
    START_DATE = '2018-01-01'

    df = read_postgres(SYMBOL, start_date=START_DATE)
    df = atr(df)
    data = to_sequence(df, SEQUENCE_LEN)
    trend, stationary = split_trending_stationary(data, spread=SPREAD)

    trend_generator = make_generator_model()
    stationary_generator = make_generator_model()

    trend_discriminator = make_discriminator_model()
    stationary_discriminator = make_discriminator_model()

    noise = tf.random.normal([1, 32])

    test = trend_generator(noise, training=False)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     trend_generator=trend_generator,
                                     stationary_generator=stationary_generator,
                                     trend_discriminator=trend_discriminator,
                                     stationary_discriminator=stationary_discriminator)

    
    noise_dim = 100
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_trend = trend_generator(noise, training=True)
            real_trend = trend_discriminator(images, training=True)
            fake_trend = trend_discriminator(generated_trend, training=True)
            trend_gen_loss = generator_loss(fake_trend)
            trend_disc_loss = discriminator_loss(real_trend, fake_trend)

            gradients_of_trend_generator = gen_tape.gradient(trend_gen_loss, trend_generator.trainable_variables)
            gradients_of_trend_discriminator = disc_tape.gradient(trend_disc_loss, trend_discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_trend_generator, trend_generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_trend_discriminator, trend_discriminator.trainable_variables))


        # log_param('noise', randint(0, 100))
        # log_metric('foo', random())
