import torch
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D, DilatedConv2D, GlobalConv2D
from inferno.extensions.layers.sampling import AnisotropicPool2D, AnisotropicUpsample2D, Upsample
from .base import UNetSkeleton, Xcoder, XcoderDilated

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
                 dil_conv_type, conv_type, scale_factor=2, local_conv_type=DilatedConv2D, activation=nn.ReLU(inplace=True), use_BN=True):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      dil_conv_type=dil_conv_type,
                                      dilation_list=dilation_list,
                                      pre_conv=get_pooler(scale_factor),
                                      local_conv_type=local_conv_type,
                                      activation=activation,
                                      use_BN=use_BN)


class Decoder(Xcoder):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type, scale_factor=2):
        super(Decoder, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type,
                                      post_conv=get_sampler(scale_factor))
                                      
class Decoder2(XcoderDilated):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list, dil_conv_type, conv_type, scale_factor=2,
                           local_conv_type=DilatedConv2D, activation=nn.ReLU(inplace=True), use_BN=True):
        super(Decoder2, self).__init__(in_channels, out_channels, kernel_size,
                                      conv_type=conv_type, 
                                      dil_conv_type=dil_conv_type,
                                      dilation_list=dilation_list,
                                      post_conv=get_sampler(scale_factor),
                                      local_conv_type=local_conv_type,
                                      activation=activation,
                                      use_BN=use_BN)

class Base(XcoderDilated):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list, dil_conv_type, conv_type, scale_factor=2, 
                        local_conv_type=DilatedConv2D, activation=nn.ReLU(inplace=True), use_BN=True):
        super(Base, self).__init__(in_channels, out_channels, kernel_size,
                                   conv_type=conv_type,
                                   dil_conv_type=dil_conv_type,
                                   dilation_list=dilation_list,
                                   pre_conv=get_pooler(scale_factor),
                                   post_conv=get_sampler(scale_factor),
                                   local_conv_type=local_conv_type,
                                   activation=activation,
                                   use_BN=use_BN)
    
    
class Output(Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Output, self).__init__(in_channels, out_channels, kernel_size)


class UNetDilated2DGlobal(UNetSkeleton):
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
                 kernel_size=5,
                 final_activation='auto',
                 first_conv_type=GlobalConv2D,
                 second_conv_type=BNReLUConv2D):
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
        first_conv_type: the type of the first convolution in the block
        second_conv_type: the same for the second one
        
        """

        assert isinstance(scale_factor, (int, list, tuple))
        #the entry can be a tuple/list for anisotropic sampling
        self.scale_factor = [scale_factor] * 2 if isinstance(scale_factor, int) else scale_factor
        assert isinstance(dilation_list, (int, list, tuple))
        if isinstance(dilation_list, int):
            dilation_list=(dilation_list,)
        self.dilation_list=dilation_list
        self.small_dilation_list=[1,3]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build encoders with proper number of feature maps
        # number of feature maps for the encoders
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth**2 
        f3e = initial_num_fmaps * fmap_growth**3 
        encoders = [
            Encoder(in_channels, f0e, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[1], scale_factor=0), #no dilation here
            Encoder(f0e, f1e, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[1], scale_factor=self.scale_factor[0]),
            Encoder(f1e, f2e, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=self.dilation_list, scale_factor=self.scale_factor[1]),
            Encoder(f2e, f3e, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[i * 2 for i in self.dilation_list],scale_factor=0)
        ]


        # Build base
        # number of base output feature maps
        f0b = initial_num_fmaps * fmap_growth**4 
        
        base = Base(f3e, f0b, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list = [i * 3 for i in self.dilation_list], scale_factor=0)

        # Build decoders
        f3d = initial_num_fmaps * fmap_growth**3 
        f2d = initial_num_fmaps * fmap_growth**2 
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            #Decoder(f0b + f3e, f3d, 3, conv_type=second_conv_type, scale_factor=0),
            Decoder2(f0b + f3e, f3d, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=self.small_dilation_list, scale_factor=0),
            #Decoder(f3d + f2e, f2d, 3, conv_type=second_conv_type, scale_factor=self.scale_factor[1]),
            Decoder2(f3d + f2e, f2d, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[i * 2 for i in self.small_dilation_list], scale_factor=self.scale_factor[1]),
            #Decoder(f2d + f1e, f1d, 3, conv_type=second_conv_type, scale_factor=self.scale_factor[0]),
            Decoder2(f2d + f1e, f1d, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[1], scale_factor=self.scale_factor[0]),
            #Decoder(f1d + f0e, f0d, 3, conv_type=second_conv_type, scale_factor=0)
            Decoder2(f1d + f0e, f0d, kernel_size=kernel_size, dil_conv_type=first_conv_type, conv_type=second_conv_type, dilation_list=[1], scale_factor=0)
        ]

        # Build output
        output = Output(f0d, out_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(UNetDilated2DGlobal, self).__init__(encoders=encoders,
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
        output = super(UNetDilated2DGlobal, self).forward(input_)
        if reshape_to_3d:
            b, c, _0, _1 = list(output.size())
            output = output.view(b, c, 1, _0, _1)
        return output
