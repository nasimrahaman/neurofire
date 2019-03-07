import torch
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D, DilatedConvELU2D
from inferno.extensions.layers.sampling import AnisotropicPool2D, AnisotropicUpsample2D, Upsample
from .base import UNetSkeleton, Xcoder, XcoderDilated
# from skunkworks.models.attention import SpatialAttentionELU2D

CONV_TYPES = {'vanilla': ConvELU2D,
              'conv_bn': BNReLUConv2D}
              # 'attention': SpatialAttentionELU2D}

def get_pooler(scale_factor):
    assert isinstance(scale_factor, (int, list, tuple))
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == 2
        assert scale_factor[0] == 1
        # we need to make sure that the scale factor conforms with the single value
        # that AnisotropicPool expects
        pooler = AnisotropicPool2D(downscale_factor=scale_factor[1])
    else:
        if scale_factor > 0:
            pooler = nn.MaxPool2d(kernel_size=1 + scale_factor,
                                   stride=scale_factor,
                                   padding=1)
        else:
            pooler = None
    return pooler


def get_sampler(scale_factor):
    assert isinstance(scale_factor, (int, list, tuple))
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == 2
        # we need to make sure that the scale factor conforms with the single value
        # that AnisotropicPool expects
        assert scale_factor[0] == 1
        sampler = AnisotropicUpsample2D(scale_factor=scale_factor[1])
    else:
        if scale_factor > 0:
            sampler = Upsample(scale_factor=scale_factor)
        else:
            sampler = None
    return sampler


class Encoder(XcoderDilated):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list, 
                 dil_conv_type=DilatedConvELU2D, conv_type=ConvELU2D, scale_factor=2):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      dil_conv_type=dil_conv_type,
                                      dilation_list=dilation_list,
                                      pre_conv=get_pooler(scale_factor))


class Decoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type=ConvELU2D, scale_factor=2):
        super(Decoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      post_conv=get_sampler(scale_factor))


class Base(XcoderDilated):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list, dil_conv_type=DilatedConvELU2D, conv_type=ConvELU2D, scale_factor=2):
        super(Base, self).__init__(in_channels, out_channels, kernel_size,
                                   conv_type=conv_type,
                                   dil_conv_type=dil_conv_type,
                                   dilation_list=dilation_list,
                                   pre_conv=get_pooler(scale_factor),
                                   post_conv=get_sampler(scale_factor))

    
    
class Output(Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Output, self).__init__(in_channels, out_channels, kernel_size)


class UNetDilated2D5l(UNetSkeleton):
    """
    2D U-Net architecture.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 dilation_list,
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
        dilation_list a list of dilations for convolutions applied parallely 
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolution type used (default: 'van[i * 5 for i in my_list]illa')
        """

        assert conv_type_key in CONV_TYPES, conv_type_key
        conv_type = CONV_TYPES[conv_type_key]
        assert isinstance(scale_factor, (int, list, tuple))
        self.scale_factor = [scale_factor] * 3 if isinstance(scale_factor, int) else scale_factor
        assert len(self.scale_factor) == 3
        #the entry can be a tuple/list for anisotropic sampling
        assert all(isinstance(sfactor, (int, list, tuple)) for sfactor in self.scale_factor)
        assert isinstance(dilation_list, (int, list, tuple))
        if isinstance(dilation_list, int):
            dilation_list=(dilation_list,)
        self.dilation_list=dilation_list
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build encoders with proper number of feature maps
        # number of feature maps for the encoders
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth**2 
        f3e = initial_num_fmaps * fmap_growth**3 
        encoders = [
            Encoder(in_channels, f0e, 3, dilation_list=[1], conv_type=conv_type, scale_factor=0), #no dilation here
            # Encoder(in_channels, f0e, 3, conv_type=SpatialAttentionELU2D, scale_factor=self.scale_factor[0]),
            Encoder(f0e, f1e, 3, dilation_list=self.dilation_list, conv_type=conv_type, scale_factor=self.scale_factor[0]),
            Encoder(f1e, f2e, 3, dilation_list=[i * 2 for i in self.dilation_list], conv_type=conv_type, scale_factor=0),
            Encoder(f2e, f3e, 3, dilation_list=[i * 3 for i in self.dilation_list], conv_type=conv_type, scale_factor=0)
        ]


        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**4 
        
        base = Base(f3e, f0b, 3, conv_type=conv_type, dilation_list = self.dilation_list, scale_factor=self.scale_factor[2])

        # Build decoders
        f3d = initial_num_fmaps * fmap_growth**3 
        f2d = initial_num_fmaps * fmap_growth**2 
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            Decoder(f0b + f3e, f3d, 3, conv_type=conv_type, scale_factor=0),
            # Decoder(f0b + f2e, f2d, 3, conv_type=SpatialAttentionELU2D, scale_factor=self.scale_factor[2]),
            Decoder(f3d + f2e, f2d, 3, conv_type=conv_type, scale_factor=0),
            Decoder(f2d + f1e, f1d, 3, conv_type=conv_type, scale_factor=self.scale_factor[0]),
            Decoder(f1d + f0e, f0d, 3, conv_type=conv_type, scale_factor=0)
        ]

        # Build output
        output = Output(f0d, out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(UNetDilated2D5l, self).__init__(encoders=encoders,
                                     base=base,
                                     decoders=decoders,
                                     output=output,
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
        output = super(UNetDilated2D5l, self).forward(input_)
        if reshape_to_3d:
            b, c, _0, _1 = list(output.size())
            output = output.view(b, c, 1, _0, _1)
        return output
