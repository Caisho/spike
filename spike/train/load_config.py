import os
import yaml


def load_configs():
    train_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(train_dir, 'configs.yml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


path = os.path.join(os.getcwd(),'spike/training/configs.yml')