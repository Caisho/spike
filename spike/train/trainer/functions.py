import os
import logging
import wandb
import datetime
import tensorflow as tf
from train.trainer.callback import CheckpointCallback


def start_training(
        train_config,
        ckpt_config,
        trend_dataset,
        stationary_dataset,
        generator_model,
        discriminator_model,
        generator_optimizer,
        discriminator_optimizer,
        generator_loss,
        discriminator_loss):

    generator_model, discriminator_model, generator_optimizer, discriminator_optimizer \
        = (restore_checkpoint(
            ckpt_config=ckpt_config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer))

    _train_loop(
        train_config=train_config,
        ckpt_config=ckpt_config,
        dataset=trend_dataset,
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss)


def restore_checkpoint(
        ckpt_config,
        generator_model,
        discriminator_model,
        generator_optimizer,
        discriminator_optimizer):
    logger = logging.getLogger(__name__)

    ckpt = tf.train.Checkpoint(
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer)

    ckpt_mgr = tf.train.CheckpointManager(
                checkpoint=ckpt,
                directory=ckpt_config['ckpt_path'],
                max_to_keep=ckpt_config['max_to_keep'])

    # TODO understand status.assert_consumed() for adam so it can be used here
    latest_ckpt = ckpt_mgr.restore_or_initialize()
    if latest_ckpt:
        logger.info('Restored model and optimizer from latest checkpoint')
    else:
        logger.info('Initialized model and optimizer from scratch')
    return generator_model, discriminator_model, generator_optimizer, discriminator_optimizer


@tf.function
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


def _train_one_epoch(
        train_config,
        dataset,
        generator_model,
        discriminator_model,
        generator_optimizer,
        discriminator_optimizer,
        generator_loss,
        discriminator_loss):
    logger = logging.getLogger(__name__)

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


def _train_loop(
        train_config,
        ckpt_config,
        dataset,
        generator_model,
        discriminator_model,
        generator_optimizer,
        discriminator_optimizer,
        generator_loss,
        discriminator_loss):
    logger = logging.getLogger(__name__)
    logger.info('Logging training config params to wandb')
    wandb.init(config=train_config, sync_tensorboard=True)

    # create checkpoint train
    ckpt = tf.train.Checkpoint(
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer)
    ckpt_mgr = tf.train.CheckpointManager(
            checkpoint=ckpt,
            directory=ckpt_config['ckpt_path'],
            max_to_keep=ckpt_config['max_to_keep'])
    callback = CheckpointCallback(ckpt_config, train_config['num_epochs'])

    # create tensorboard train logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(train_config['tensorboard_path'], 'train', current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir, name=train_config['model'])

    for epoch in range(train_config['num_epochs']):
        logger.info(f'Running Epoch {epoch}')
        gen_loss, disc_loss = _train_one_epoch(
            train_config=train_config,
            dataset=dataset,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss)

        # write to tensorboard train logs
        with train_summary_writer.as_default():
            tf.summary.scalar('generator_loss', gen_loss, step=epoch)
            tf.summary.scalar('discriminator_loss', disc_loss, step=epoch)

        # save checkpoints
        callback.on_epoch_end(epoch, ckpt_mgr)

    # log metrics to wandb
    wandb.log({'generator_loss': gen_loss.numpy(), 'discriminator_loss': disc_loss.numpy()})
    # save checkpoint files to wandb
    wandb.save(os.path.join(ckpt_config['ckpt_path'], '*ckpt*'))
