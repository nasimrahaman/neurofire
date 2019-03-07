# Adapted from https://github.com/xlliu7/hed.pytorch

import torch.nn as nn
import torch
import torch.nn.functional as F
from inferno.extensions.layers.convolutional import ConvELU2D, ConvELU3D
from inferno.extensions.layers.convolutional import Conv2D, Conv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample


# TODO for 3d it might be benefitial to go back to 2 convs for the first 2 layers
# NOTE we use 3 convolutions for all blocks
# in the initial implementations, the first 2 blocks only
# consist of 2 convolutions
class BaseHEDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type1, conv_type2, conv_type3,
                 dilation=1, kernel=3):
        super(BaseHEDBlock, self).__init__()
        self.conv = nn.Sequential(conv_type1(in_channels, out_channels, kernel),
                                  conv_type2(out_channels, out_channels, kernel),
                                  conv_type3(out_channels, out_channels, kernel))

    def forward(self, x):
        return self.conv(x)


class DefaultHEDBlock(BaseHEDBlock):
    def __init__(self, in_channels, out_channels, dilation=1, stride=3):
        super(DefaultHEDBlock, self).__init__(in_channels, out_channels, ConvELU2D, ConvELU2D, ConvELU2D)


class DefaultHEDBlock3D(BaseHEDBlock):
    def __init__(self, in_channels, out_channels, dilation=1, stride=3):
        super(DefaultHEDBlock3D, self).__init__(in_channels, out_channels, ConvELU3D, ConvELU3D, ConvELU3D)


class Upsampling3d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling3d, self).__init__()
        self.sample = nn.Upsample(scale_factor=scale_factor, mode='trilinear',
                                  align_corners=False)

    def forward(self, x):
        return self.sample(x)


class Upsampling2d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling2d, self).__init__()
        self.sample = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                                  align_corners=False)

    def forward(self, x):
        return self.sample(x)


# Wrapper to allow calling AnisotropicPool with stride
class PoolAniso(AnisotropicPool):
    def __init__(self, scale_factor, stride):
        super(PoolAniso, self).__init__(scale_factor)


class HED(nn.Module):
    block_types = {'default': DefaultHEDBlock,
                   'default3d': DefaultHEDBlock3D}
    sampling_types = {'default': (nn.MaxPool2d, Upsampling2d),
                      'default3d': (nn.MaxPool3d, Upsampling3d),
                      'anisotropic': (PoolAniso, AnisotropicUpsample)}
    output_types = {'default': Conv2D,
                    'default3d': Conv3D}

    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 scale_factor=2,
                 block_type_key='default',
                 output_type_key='default',
                 sampling_type_key='default',
                 rescale_outputs=True):

        # scale factors can be list or single value
        assert isinstance(scale_factor, (int, list, tuple))
        if isinstance(scale_factor, (list, tuple)):
            assert len(scale_factor) == 4
            self.scale_factor = scale_factor
        else:
            self.scale_factor = 4 * (scale_factor,)

        # block types can be single key or list of keys
        assert isinstance(block_type_key, (str, list, tuple))
        if isinstance(block_type_key, (list, tuple)):
            assert len(block_type_key) == 5
            assert all(bkt in self.block_types for bkt in block_type_key), block_type_key
            self.block_type_key = block_type_key
        else:
            assert block_type_key in self.block_types, block_type_key
            self.block_type_key = 5 * (block_type_key,)

        assert output_type_key in self.output_types, output_type_key
        output_type = self.output_types[output_type_key]

        # sampling types can be single key or list of keys
        if isinstance(sampling_type_key, (list, tuple)):
            assert len(sampling_type_key) == 4
            assert all(skt in self.sampling_types
                       for skt in sampling_type_key), sampling_type_key
            self.sampling_type_key = sampling_type_key
        else:
            assert sampling_type_key in self.sampling_types, sampling_type_key
            self.sampling_type_key = 4 * (sampling_type_key,)

        super(HED, self).__init__()
        # calculate number of features for all levels
        f0 = initial_num_fmaps
        f1 = initial_num_fmaps * fmap_growth
        f2 = initial_num_fmaps * fmap_growth**2
        f3 = initial_num_fmaps * fmap_growth**3

        # convolutional blocks
        self.conv0 = self.block_types[self.block_type_key[0]](in_channels, f0)
        self.conv1 = self.block_types[self.block_type_key[1]](f0, f1)
        self.conv2 = self.block_types[self.block_type_key[2]](f1, f2)
        self.conv3 = self.block_types[self.block_type_key[3]](f2, f3)
        # we don't change the number of feats in the last conv block
        self.conv4 = self.block_types[self.block_type_key[4]](f3, f3)

        # poolers
        sample_type0 = self.sampling_types[self.sampling_type_key[0]]
        self.pool0 = sample_type0[0](self.scale_factor[0], stride=self.scale_factor[0])

        sample_type1 = self.sampling_types[self.sampling_type_key[1]]
        self.pool1 = sample_type1[0](self.scale_factor[1], stride=self.scale_factor[1])

        sample_type2 = self.sampling_types[self.sampling_type_key[2]]
        self.pool2 = sample_type2[0](self.scale_factor[2], stride=self.scale_factor[2])

        sample_type3 = self.sampling_types[self.sampling_type_key[3]]
        self.pool3 = sample_type3[0](self.scale_factor[3], stride=self.scale_factor[3])

        self.out0 = output_type(f0, out_channels, 1)
        self.out1 = output_type(f1, out_channels, 1)
        self.out2 = output_type(f2, out_channels, 1)
        self.out3 = output_type(f3, out_channels, 1)
        self.out4 = output_type(f3, out_channels, 1)
        # 6 is the fusion layer -> 5 * out_channels
        self.out5 = output_type(5*out_channels, out_channels, 1)

        self.upsample0 = sample_type0[1](scale_factor=self.scale_factor[0])
        self.upsample1 = sample_type1[1](scale_factor=self.scale_factor[1])
        self.upsample2 = sample_type2[1](scale_factor=self.scale_factor[2])
        self.upsample3 = sample_type3[1](scale_factor=self.scale_factor[3])
        self.rescale_outputs = rescale_outputs

    def forward(self, x):

        # apply convolutions and poolings
        conv0 = self.conv0(x)

        conv1 = self.pool0(conv0)
        conv1 = self.conv1(conv1)

        conv2 = self.pool1(conv1)
        conv2 = self.conv2(conv2)

        conv3 = self.pool2(conv2)
        conv3 = self.conv3(conv3)

        conv4 = self.pool3(conv3)
        conv4 = self.conv4(conv4)

        if self.rescale_outputs:

            # make side output
            # NOTE we may have different pooling schemes, so we need to apply
            # all the samplers in a chain
            out0 = self.out0(conv0)
            out1 = self.upsample0(self.out1(conv1))
            out2 = self.upsample1(self.upsample0(self.out2(conv2)))
            out3 = self.upsample2(self.upsample1(self.upsample0(self.out3(conv3))))
            out4 = self.upsample3(self.upsample2(self.upsample1(self.upsample0(self.out4(conv4)))))

            # make fusion output
            out5 = self.out5(torch.cat((out0, out1, out2, out3, out4), 1))

        else:

            # make side output
            # NOTE we may have different pooling schemes, so we need to apply
            # all the samplers in a chain
            out0 = self.out0(conv0)
            out1 = self.out1(conv1)
            out2 = self.out2(conv2)
            out3 = self.out3(conv3)
            out4 = self.out4(conv4)

            upsampled1 = self.upsample0(out1)
            upsampled2 = self.upsample1(self.upsample0(out2))
            upsampled3 = self.upsample2(self.upsample1(self.upsample0(out3)))
            upsampled4 = self.upsample3(self.upsample2(self.upsample1(self.upsample0(out4))))

            # make fusion output
            out5 = self.out5(torch.cat((out0, upsampled1, upsampled2, upsampled3, upsampled4), 1))

        # apply activations
        # TODO enable different activations
        out0 = F.sigmoid(out0)
        out1 = F.sigmoid(out1)
        out2 = F.sigmoid(out2)
        out3 = F.sigmoid(out3)
        out4 = F.sigmoid(out4)
        out5 = F.sigmoid(out5)

        # we return first, because it is usually the one used for stuff
        return out5, out0, out1, out2, out3, out4
