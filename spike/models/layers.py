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
)


def generator():
    model = Sequential()
    model.add(Dense(4096, input_shape=(32,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((128, 32)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(16, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(8, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(4, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    return model


def discriminator():
    model = Sequential()
    model.add(Conv1D(64, 3, strides=2, input_shape=[512, 4], padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv1D(128, 3, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))
    return model
