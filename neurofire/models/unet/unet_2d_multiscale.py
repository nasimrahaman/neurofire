import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D
from .base import UNetSkeletonMultiscale, Xcoder

CONV_TYPES = {'vanilla': ConvELU2D,
              'conv_bn': BNReLUConv2D}


class Encoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type=ConvELU2D, scale_factor=2):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      pre_output=nn.MaxPool2d(kernel_size=1 + scale_factor,
                                                              stride=scale_factor,
                                                              padding=1))


class Decoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type=ConvELU2D, scale_factor=2):
        super(Decoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      pre_output=nn.Upsample(scale_factor=scale_factor))


class Base(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type=ConvELU2D):
        super(Base, self).__init__(in_channels, out_channels, kernel_size,
                                   conv_type=conv_type,
                                   pre_output=None)


class Output(Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Output, self).__init__(in_channels, out_channels, kernel_size)


class UNet2DMultiscale(UNetSkeletonMultiscale):
    """
    2D U-Net architecture.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 scale_factor=2,
                 final_activation='auto',
                 conv_type_key='vanilla'):
        """
        Parameter:
        ----------
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        initial_num_fmaps (int): number of feature maps of the first layer
        fmap_growth (int): growth factor of the feature maps; the number of feature maps
        in layer k is given by initial_num_fmaps * fmap_growth**k
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolution type used (default: 'vanilla')
        """

        assert conv_type_key in CONV_TYPES, conv_type_key
        conv_type = CONV_TYPES[conv_type_key]
        assert isinstance(scale_factor, (int, list, tuple))
        self.scale_factor = [scale_factor] * 3 if isinstance(scale_factor, int) else scale_factor
        assert len(self.scale_factor) == 3
        assert all(isinstance(sfactor, int) for sfactor in self.scale_factor)

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build encoders with proper number of feature maps
        # number of feature maps for the encoders
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth**2
        encoders = [
            Encoder(in_channels, f0e, 3, conv_type=conv_type, scale_factor=self.scale_factor[0]),
            Encoder(f0e, f1e, 3, conv_type=conv_type, scale_factor=self.scale_factor[1]),
            Encoder(f1e, f2e, 3, conv_type=conv_type, scale_factor=self.scale_factor[2])
        ]

        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**3
        base = Base(f2e, f0b, 3)

        # Build decoders
        f2d = initial_num_fmaps * fmap_growth**2
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            Decoder(f0b + f2e + out_channels, f2d, 3, conv_type=conv_type, scale_factor=self.scale_factor[2]),
            Decoder(f2d + f1e + out_channels, f1d, 3, conv_type=conv_type, scale_factor=self.scale_factor[1]),
            Decoder(f1d + f0e + out_channels, f0d, 3, conv_type=conv_type, scale_factor=self.scale_factor[0])
        ]


        # Build decoders
        output_0 = Output(f0d, out_channels, 3)
        output_1 = Output(f1d, out_channels, 3)
        output_2 = Output(f2d, out_channels, 3)
        output_3 = Output(f0b, out_channels, 3)
        predictors = [output_0, output_1, output_2, output_3]

        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(UNet2DMultiscale, self).__init__(encoders=encoders,
                                     base=base,
                                     decoders=decoders,
                                     predictors=predictors,
                                     final_activation=final_activation)

    def forward(self, input_):
        # some loaders are usually 3D, so we reshape if necessary
        if input_.dim() == 5:
            reshape_to_3d = True
            b, c, _0, _1, _2 = list(input_.size())
            assert _0 == 1, "%i" % _0
            input_ = input_.view(b, c * _0, _1, _2)
        else:
            reshape_to_3d = False
        output = super(UNet2DMultiscale, self).forward(input_)
        if reshape_to_3d:
            b, c, _0, _1 = list(output.size())
            output = output.view(b, c, 1, _0, _1)
        return output
