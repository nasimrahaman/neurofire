from __future__ import print_function

import neurofire
from neurofire.datasets.isbi2012.loaders import get_isbi_loader
from neurofire.models import UNet2D

import torch
from torch.autograd import Variable

# TODO affinities + malis
def train(use_gpu=False):
    model = UNet2D(1, 1, n_scale=2) # 1 grayscale input channel, 1 output channel (membrane probability)

    train_loader, validate_loader = get_isbi_loader('./data_config.yml')

    trainer = Trainer(model)
    trainer.build_criterion('SorensenDiceLoss')
    trainer.build_optimizer('Adam')

    # TODO need validation
    #trainer.build_metric('')
    #trainer.validate_every((1, 'epochs'))

    trainer.save_every((1000, 'iterations'))
    trainer.set_max_num_iterations(1e4)
    # TODO Tensorboard logger

    trainer.bind_loader('train', train_loader)
    # trainer.bind_loader('test', validate_loader)

    if use_gpu:
        trainer.cuda()

    trainer.fit()


if __name__ == '__main__':
    train()
