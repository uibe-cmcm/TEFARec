import argparse
import inspect

import torch


class Config:
    device = torch.device("cuda:0")
    train_epochs = 100
    batch_size = 512
    learning_rate = 0.004
    l2_regularization = 5e-4  # 权重衰减程度
    learning_rate_decay = 0.99
    embed_size = 24
    model_file = 'best_model.pt'
    data_name = 'dm'
    # data_name = 'yelp'
    dropout_prob = 0.5

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
