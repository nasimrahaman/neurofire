import torch
import torch.nn as nn


class Xcoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type,
                 pre_conv=None, post_conv=None):
        super(Xcoder, self).__init__()
        assert out_channels % 2 == 0
        # make sure that conv-type
        assert isinstance(conv_type, type)
        # the in-channels we get from the top-level / bottom level layer + skip connections
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.conv1 = conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.conv2 = conv_type(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.pre_conv = pre_conv
        self.post_conv = post_conv

    #
    # noinspection PyCallingNonCallable
    def forward(self, input_):
        input_ = input_ if self.pre_conv is None else self.pre_conv(input_)
        out = self.conv1(input_)
        out = self.conv2(out)
        out = out if self.post_conv is None else self.post_conv(out)
        return out


class XcoderResidual(Xcoder):
    """
    Inspired by arXiv:1706.00120

    """

    def __init__(self, *super_args, **super_kwargs):
        super(XcoderResidual, self).__init__(*super_args, **super_kwargs)

        # Add initial 2D convolution:
        if isinstance(self.kernel_size, int):
            self.kernel_size_2d = (1, self.kernel_size, self.kernel_size)
        else:
            assert isinstance(self.kernel_size, tuple)
            assert len(self.kernel_size) == 3
            self.kernel_size_2d = (1, self.kernel_size[1], self.kernel_size[2])

        # change block 1 to 2d conv
        self.conv1 = self.conv_type(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=self.kernel_size_2d)

        # add an additional conv layer
        self.conv3 = self.conv_type(in_channels=self.out_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=self.kernel_size)

    def forward(self, input_):
        input_ = input_ if self.pre_conv is None else self.pre_conv(input_)
        conv1_out = self.conv1(input_)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        # Add skip connection:
        out = conv1_out + conv3_out

        if self.post_conv is not None:
            out = self.post_conv(out)

        return out

class XcoderDilated (nn.Module):
	#applies dilated convolutions in parallel, cats the results, and applies one more conv
    def __init__(self, in_channels, out_channels, kernel_size, dil_conv_type,
                 conv_type, dilation_list, pre_conv=None, post_conv=None, **kwargs):
        super(XcoderDilated, self).__init__()
        assert out_channels % 2 == 0
        assert isinstance(conv_type, type)
        assert isinstance(dilation_list, (list, tuple)) #the list of dilations applied in parralel
        # the in-channels we get from the top-level / bottom level layer + skip connections
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dil_conv_type = dil_conv_type
        self.conv_type = conv_type
        self.dilation_list = dilation_list
        self.pre_conv = pre_conv
        self.post_conv = post_conv
        self.convs=nn.ModuleList()
        for dilation in dilation_list:
            self.convs.append(dil_conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               dilation=dilation, **kwargs))
        self.conv2 = conv_type(in_channels=self.out_channels*len(dilation_list),
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size)
        self.pre_conv = pre_conv
        self.post_conv = post_conv

    #
    # noinspection PyCallingNonCallable
    def forward(self, input_):
        input_ = input_ if self.pre_conv is None else self.pre_conv(input_)
        outputs = []
        for conv1 in self.convs:
            outputs.append(conv1(input_))
        out = torch.cat(outputs, 1)
        out = self.conv2(out)
        out = out if self.post_conv is None else self.post_conv(out)
        return out


class UNetSkeleton(nn.Module):
    def __init__(self, encoders, base, decoders, output, final_activation=None):
        super(UNetSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        assert len(encoders) == len(decoders), "%i, %i" % (len(encoders), len(decoders))
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.base = base
        self.output = output
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def forward(self, input_):

        x = input_
        encoder_out = []
        # apply encoders and remember their outputs
        for encoder in self.encoders:
            x = encoder(x)
            encoder_out.append(x)

        x = self.base(x)

        # apply decoders
        max_level = len(self.decoders) - 1
        for level, decoder in enumerate(self.decoders):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((x, encoder_out[max_level - level]), 1))

        x = self.output(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
