import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Reshape,
    Conv1D,
    UpSampling1D,
    Dropout,
    Flatten,
    LeakyReLU,
    LSTM,
)


# def generator(input_dim, noise_dim):
#     model = Sequential()
#     model.add(Conv1D(input_dim*pow(2, 3), 3, padding='same', input_shape=[noise_dim, 1]))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(input_dim*pow(2, 2), 3, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(input_dim*pow(2, 1), 3, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(UpSampling1D(size=2))
#     model.add(Conv1D(input_dim*pow(2, 0), 3, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     return model


# def discriminator(seq_len, input_dim):
#     model = Sequential()
#     model.add(Conv1D(input_dim*pow(2, 0), 3, strides=1, input_shape=[seq_len, input_dim], padding='same'))
#     model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(Conv1D(input_dim*pow(2, 1), 3, strides=2, padding='same'))
#     model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(Conv1D(input_dim*pow(2, 2), 3, strides=2, padding='same'))
#     model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(Conv1D(input_dim*pow(2, 3), 3, strides=2, padding='same'))
#     model.add(Dropout(0.3))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     model.add(Flatten())
#     model.add(Dense(1))
#     return model


def generator(Z, T, noise_dim):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=[T, noise_dim]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(Z, activation=tf.keras.activations.sigmoid))
    return model


def discriminator(H, T):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=[T, H]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation=None))
    return model


def supervisor(H, T):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=[T, H]))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(H, activation=tf.keras.activations.sigmoid))
    return model
