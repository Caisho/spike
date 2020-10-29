import os
import logging
import wandb
import tensorflow as tf
from models.loss import generator_loss, discriminator_loss


def restore_checkpoint(model_name, model, optimizer, checkpoint_dir, checkpoint_suffix):
    pass


# TODO understand why tf.function has error
# @tf.function
def _train_step(
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


def _train_one_epoch(train_config, dataset, generator_model, discriminator_model):
    logger = logging.getLogger(__name__)

    generator_optimizer = tf.keras.optimizers.Adam(train_config['optimizer']['learning_rate'])
    discriminator_optimizer = tf.keras.optimizers.Adam(train_config['optimizer']['learning_rate'])

    for batch_data in dataset:
        gen_loss, disc_loss = _train_step(
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
    return gen_loss, disc_loss


def train_loop(train_config, ckpt_config, dataset, generator_model, discriminator_model):
    logger = logging.getLogger(__name__)

    # TODO checkpoint, tensorboard run name, display sample generated dataset in tensorboard?

    # create tensorboard train logs
    train_log_dir = os.path.join(train_config['tensorboard_path'], 'tb_logs', 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir, name=train_config['model_name'])

    for epoch in range(train_config['num_epochs']):
        logger.info(f'Running Epoch {epoch}')
        gen_loss, disc_loss = _train_one_epoch(train_config, dataset, generator_model, discriminator_model)

        # write to tensorboard train logs
        with train_summary_writer.as_default():
            tf.summary.scalar('generator_loss', gen_loss, step=epoch)
            tf.summary.scalar('discriminator_loss', disc_loss, step=epoch)

    # log metrics to wandb
    wandb.log({'generator_loss': gen_loss.numpy(), 'discriminator_loss': disc_loss.numpy()})
