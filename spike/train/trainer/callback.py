import logging
from tensorflow.keras.callbacks import Callback


class CheckpointCallback(Callback):

    def __init__(self, ckpt_config, num_epoch):
        super().__init__()

        self.ckpt_config = ckpt_config
        self.num_epoch = num_epoch
        self.logger = logging.getLogger(__name__)

    def _should_save_epoch(self, epoch):
        first_epoch = (epoch == 1)
        reached_next_n_epoch = ((epoch % self.ckpt_config['ckpt_step']) == 0)
        final_epoch = (epoch >= self.num_epoch)
        return first_epoch or reached_next_n_epoch or final_epoch

    def on_epoch_end(self, epoch, ckpt_mgr):
        epoch += 1
        if self._should_save_epoch(epoch):
            save_path = ckpt_mgr.save()
            self.logger.info(f'Epoch {epoch} has been saved to {save_path}')
