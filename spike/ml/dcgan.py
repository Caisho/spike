import tensorflow as tf
from tensorflow.keras import layers
from spike.etl.extract import read_postgres
from spike.etl.transform import atr, to_sequence, split_trending_stationary


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


df = read_postgres('eurusd_m1', start_date='2018-01-01')
df = atr(df)
data = to_sequence(df, 50)
trend, stationary = split_trending_stationary(data, spread=0.0003)

trend_generator = make_generator_model()
stationary_generator = make_generator_model()

trend_discriminator = make_discriminator_model()
stationary_discriminator = make_discriminator_model()

noise = tf.random.normal([1,32])

test = trend_generator(noise, training=False)

