import torch.nn as nn
from inferno.extensions.layers.sampling import Upsample
from .base import UNetSkeletonMultiscale
from ..unet.unet_2d import Encoder, Decoder, Output, CONV_TYPES


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
        assert isinstance(out_channels, (list, int, tuple))
        if isinstance(out_channels, int):
            out_channels = 4 * [out_channels]
        else:
            assert len(out_channels) == 4

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels[0]

        # Build encoders with proper number of feature maps
        # number of feature maps for the encoders
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth**2
        encoders = [
            Encoder(in_channels, f0e, 3, conv_type=conv_type, scale_factor=0),
            Encoder(f0e, f1e, 3, conv_type=conv_type, scale_factor=self.scale_factor[0]),
            Encoder(f1e, f2e, 3, conv_type=conv_type, scale_factor=self.scale_factor[1])
        ]

        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**3
        # NOTE: we do not upsample in the base
        base = Encoder(f2e, f0b, 3, conv_type=conv_type, scale_factor=self.scale_factor[2])

        # Build decoders
        f2d = initial_num_fmaps * fmap_growth**2
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        # NOTE: We need extra samplers for the multi-scale stuff here,
        # so we don't sample in the decoders
        decoders = [
            Decoder(f0b + f2e + out_channels[3], f2d, 3, conv_type=conv_type, scale_factor=0),
            Decoder(f2d + f1e + out_channels[2], f1d, 3, conv_type=conv_type, scale_factor=0),
            Decoder(f1d + f0e + out_channels[1], f0d, 3, conv_type=conv_type, scale_factor=0)
        ]

        samplers = [Upsample(scale_factor=sf) for sf in reversed(self.scale_factor)]

        # Build outputs
        output_0 = Output(f0d, out_channels[0], 3)
        output_1 = Output(f1d, out_channels[1], 3)
        output_2 = Output(f2d, out_channels[2], 3)
        output_3 = Output(f0b, out_channels[3], 3)
        predictors = [output_0, output_1, output_2, output_3]

        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if self.out_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(UNet2DMultiscale, self).__init__(encoders=encoders,
                                               base=base,
                                               decoders=decoders,
                                               predictors=predictors,
                                               samplers=samplers,
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
            outsize = [list(out.size()) for out in output]
            output = [out.view(osize[0], osize[1], 1, osize[2], osize[3])
                      for out, osize in zip(output, outsize)]
        return output
