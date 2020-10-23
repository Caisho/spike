from datetime import timedelta
import logging
import os
import time

import tensorflow as tf

from devtools.train.load_configs import load_configs

from .config import CheckpointConfig
from .callback import CheckpointCallback

MAX_MODELS_TO_KEEP = 5

CONFIGS = load_configs()['trainer']


def start_distributed_training(
        model_class, model_name, epochs, validation_freq, output_dir):
    """Build a model and train it using distributed strategy

    Explanation of how this function works.

    model_class: model_class is the Class that has 1 required methods: model_class.compile().
        - model_class.compile() returns the TrainingConfig namedtuple and creates your model inside.
            For more information about TrainingConfig, refer to devtools/train/trainer/config.py.


    args:
        model_class - the main class of your model. it needs to have a .compile()
                    method that returns TrainingConfig and a .create_model() method
                    that returns the keras/tensorflow model.
        model_name - the model name to create
        epochs - number of epochs that should be trained
        validation_freq - how often to perform validation
        output_dir - folder where checkpoint and model file is saved
    """

    strategy = tf.distribute.MirroredStrategy()
    logger = logging.getLogger(__name__)
    logger.info('Number of devices for distributed training: %s',
                strategy.num_replicas_in_sync)

    training_config = None
    ckpt_config = None
    with strategy.scope():
        training_config = model_class.compile()
        ckpt_config = restore_checkpoint(
            model_name,
            training_config.model,
            training_config.optimizer,
            output_dir,
            training_config.checkpoint_suffix)

    train_loop(strategy, training_config, ckpt_config, epochs, validation_freq)


def _get_checkpoint_path(checkpoint_dir, model_name, checkpoint_suffix):
    ckpt_path = os.path.join(
        checkpoint_dir, model_name, "checkpoint" + checkpoint_suffix)
    os.makedirs(ckpt_path, exist_ok=True)
    return ckpt_path



def restore_checkpoint(
        model_name, model, optimizer, checkpoint_dir, checkpoint_suffix):
    """Restore the last checkpoint

    args:
        model_name - model name string
        model - the tensorflow model instance
        optimizer - the optimizer determine how to compute the loss gradient
        checkpoint_dir - folder path where checkpoint is saved
        checkpoint_suffix - the model filename suffix for export
    """
    logger = logging.getLogger(__name__)

    ckpt_path = _get_checkpoint_path(
        checkpoint_dir, model_name, checkpoint_suffix)
    epoch_path = _get_epoch_path(ckpt_path)
    last_epoch = 0
    if os.path.exists(epoch_path):
        with open(epoch_path) as f:
            last_epoch = int(f.read())

    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model)

    ckpt_mgr = tf.train.CheckpointManager(ckpt,
                                          ckpt_path,
                                          max_to_keep=MAX_MODELS_TO_KEEP)

    ckpt.restore(ckpt_mgr.latest_checkpoint)
    if ckpt_mgr.latest_checkpoint:
        logger.info("Restored from %s", ckpt_mgr.latest_checkpoint)
    else:
        logger.info("Initializing from scratch.")

    return CheckpointConfig(model, ckpt_mgr, ckpt_path, epoch_path, last_epoch)


@tf.function
def train_step(batch_data, global_batch_sizes, model, loss_function, optimizer):
    """
    Perform a train gradient descent step.

    args:
        batch_data - (X, Y1, Y2, ..., Yn) where X is the batch training input.
                     Y1, ..., Yn is the training target output.
        global_batch_sizes - list of global batch size for each target output in
                             (X, Y1, Y2, ..., Yn)
        model - the tf model to be trained
        loss_function - a loss function that expects the (Y1, ..., Yn) as first
                        argument and the predictions as second argument
        optimizer - the loss minimizer
    """
    X, Y, keypoints = batch_data[0], batch_data[1:-1], batch_data[-1]

    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = loss_function(global_batch_sizes, Y, predictions, keypoints)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def _train_one_epoch(
        strategy, train_iter, training_config, timestamps,
        current_epoch, epoch):
    logger = logging.getLogger(__name__)

    model = training_config.model
    loss_function = training_config.loss_function
    batch_size_function = training_config.batch_size_function
    optimizer = training_config.optimizer

    optimizer.learning_rate = training_config.lr_schedule(current_epoch)

    epoch_loss = 0.0
    train_steps = training_config.train_steps
    for step in range(train_steps):
        batch_data = next(train_iter)
        step_loss = distributed_train_step(
            strategy, batch_data, model, batch_size_function, loss_function,
            optimizer)
        epoch_loss += step_loss

        eta = _track_current_eta(timestamps, step, train_steps)

        logger.info("Epoch %s/%s Step %s/%s loss: %s eta: %s",
                    current_epoch,
                    epoch,
                    step + 1,
                    train_steps,
                    step_loss,
                    eta)

    return epoch_loss / train_steps


def train_loop(strategy, training_config, ckpt_config, epoch, validation_freq):
    """
    Generic training loop for tensorflow pose estimation. Saves the checkpoints
    and model periodically.

    This loop is highly recommended to be out of strategy scope because:
    https://github.com/tensorflow/tensorflow/issues/28599
    https://github.com/tensorflow/tensorflow/issues/30850

    args:
        strategy - instance of tf.distribute.Strategy
        training_config - pose.trainer.config.Config that specify how to train
                          model
        ckpt_config - checkpoint config to resume training from last run
        epoch - number of epoch to train
        validation_freq - how often to perform validation
    """

    # create tensorboard logs
    train_log_dir = os.path.join(ckpt_config.ckpt_path, 'tb_logs', 'train')
    val_log_dir = os.path.join(ckpt_config.ckpt_path, 'tb_logs', 'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir, name='hourglass')
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    keypoint_callback = CheckpointCallback(
        training_config.model_name, ckpt_config, epoch, MAX_MODELS_TO_KEEP)

    timestamps = [time.perf_counter()]

    for e in range(ckpt_config.last_epoch, epoch):
        if CONFIGS['enable_profile_tracing']:
            tf.summary.trace_on(graph=True, profiler=True)
        current_epoch = e + 1

        # resets the dataset generators
        train_iter = iter(strategy.experimental_distribute_dataset(
            training_config.train_data))

        val_iter = iter(strategy.experimental_distribute_dataset(
            training_config.val_data))

        epoch_results = {}
        epoch_results['loss'] = _train_one_epoch(
            strategy, train_iter, training_config, timestamps,
            current_epoch, epoch)

        if _should_validate(e, epoch, validation_freq):
            epoch_results['val_loss'] = _compute_validation_loss(
                strategy, val_iter, training_config)

        # writes to tb train logger
        with train_summary_writer.as_default():  # pylint: disable=not-context-manager
            if CONFIGS['enable_profile_tracing']:
                tf.summary.trace_export(name='hourglass', step=e,
                                        profiler_outdir=os.path.join(ckpt_config.ckpt_path,
                                                                     'tb_logs'))
            tf.summary.scalar('loss', epoch_results['loss'], step=e)

        # tests if val_loss is in epoch before writing it to tb logger
        if 'val_loss' in epoch_results:
            with val_summary_writer.as_default():  # pylint: disable=not-context-manager
                tf.summary.scalar('loss', epoch_results['val_loss'], step=e)

        keypoint_callback.on_epoch_end(e, epoch_results)
