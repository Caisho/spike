import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
)
from dataset.dataset import FxDataset


def generator(Z, T, noise_dim):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=[T, noise_dim]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(Z, activation=tf.keras.activations.sigmoid))
    return model


def supervisor(H, T):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=[T, H]))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(H, activation=tf.keras.activations.sigmoid))
    return model


def supervisor_loss(H, H_hat):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(H[:, 1:, :], H_hat[:, :-1, :]))


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    t = tf.ones_like(fake_output)
    print(t.shape)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(batch_data, model, optimizer, loss):
    with tf.GradientTape() as tape:
        H_hat = model(batch_data, training=True)
        batch_loss = loss(batch_data, H_hat)
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return batch_loss


EPOCHS = 10
optimizer = tf.keras.optimizers.Adam()
dataset = FxDataset().get_dataset()[0]
model = supervisor(4, 512)
batch_loss = 0

for epoch in range(EPOCHS):
    print(f'Running Epoch {epoch}')
    for batch_data in dataset:
        batch_loss = train_step(batch_data, model, optimizer, supervisor_loss)
    print('Batch Loss = ', batch_loss)


for batch in dataset:
    ones = tf.ones_like(batch)
    loss = generator_loss(batch)
    print(loss)