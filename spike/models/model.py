import logging
import tensorflow as tf
from train.load_config import load_configs


class DcganModel():

    def __init__(self, configs):
        self.logger = logging.getLogger(__name__)
        self.config = load_configs()
        self.model_name = 'dcgan'

    def create_models(self):
        self.logger.info('Creating dcgan model')
        

    def predict(self, data):
        pass
    
    @tf.function
    def train(self, input_dir, output_dir, use_case, epochs=1):
        noise = tf.random.normal([self.config['training']['batch_size'], noise_dim])
        pass
