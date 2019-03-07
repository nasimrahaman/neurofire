from itertools import chain
import torch
import torch.nn as nn
from .hed import HED


# TODO reference to paper
class M2FCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 initial_num_fmaps,
                 fmap_growth=2,
                 scale_factor=2,
                 block_type_key='default',
                 output_type_key='default',
                 sampling_type_key='default',
                 fuse_sub_networks=False):


        assert isinstance(initial_num_fmaps, (list, tuple))
        super(M2FCN, self).__init__()
        self.heds = nn.ModuleList([HED(in_channels=in_channels,
                                       out_channels=out_channels,
                                       initial_num_fmaps=initial_f,
                                       fmap_growth=fmap_growth,
                                       scale_factor=scale_factor,
                                       block_type_key=block_type_key,
                                       output_type_key=output_type_key,
                                       sampling_type_key=sampling_type_key)
                                  for initial_f in initial_num_fmaps])

    # TODO concate all outputs instead of only fusion ?!
    # TODO consider applying hed2 iteratively
    def forward(self, x):
        out = []
        for i, hed in enumerate(self.heds):
            if i == 0:
                out.append(hed(x))
            else:
                out.append(hed(out[-1][0]))

        return tuple(chain(*out))
