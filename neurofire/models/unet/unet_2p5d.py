import torch.nn as nn
from .base import UNetSkeleton
from .unet_2d import Encoder, Decoder, Base, Output, CONV_TYPES
from inferno.extensions.layers.reshape import As2D, As3D


class UNet2p5D(UNetSkeleton):
    """
    2.5D U-Net architecture.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 z_channels,
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
        z_channels (int): number of z-channels that are taken as input channels
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
        self.z_slices = z_channels
        # Recompute in channels and out channels
        in_channels = self.z_slices * self.in_channels
        out_channels = self.z_slices * self.out_channels

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
        base = Base(f2e, f0b, 3, scale_factor=self.scale_factor[2])

        # Build decoders
        f2d = initial_num_fmaps * fmap_growth**2
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            Decoder(f0b + f2e, f2d, 3, conv_type=conv_type, scale_factor=self.scale_factor[1]),
            Decoder(f2d + f1e, f1d, 3, conv_type=conv_type, scale_factor=self.scale_factor[2]),
            Decoder(f1d + f0e, f0d, 3, conv_type=conv_type, scale_factor=0)
        ]

        # Build output
        output = Output(f0d, out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(UNet2p5D, self).__init__(encoders=encoders,
                                       base=base,
                                       decoders=decoders,
                                       output=output,
                                       final_activation=final_activation)

        # Build reshape layers
        self.as_2d = As2D(z_as_channel=True)
        self.as_3d = As3D(channel_as_z=True, num_channels_or_num_z_slices=self.out_channels)

    # noinspection PyCallingNonCallable
    def forward(self, input_):
        # Make sure input has the right shape
        assert input_.dim() == 5
        assert input_.size(2) == self.z_slices
        # We're expecting 3D inputs - so we convert to 2D
        input_2d = self.as_2d(input_)
        output_2d = super(UNet2p5D, self).forward(input_2d)
        output = self.as_3d(output_2d)
        return output
