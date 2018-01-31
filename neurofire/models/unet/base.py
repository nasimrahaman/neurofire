import torch
import torch.nn as nn

class Xcoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type, pre_output):
        super(Xcoder, self).__init__()
        assert out_channels % 2 == 0
        # make sure that conv-type
        assert isinstance(conv_type, type)
        # the in-channels we get from the top-level / bottom level layer + skip connections
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO how do the filter sizes evolve here?
        # conv1: in -> out, conv2: out -> out
        # conv1: in -> in, conv2: in -> out (???)
        self.conv1 = conv_type(in_channels=self.in_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.conv2 = conv_type(in_channels=self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size)
        self.pre_output = pre_output

    #
    # noinspection PyCallingNonCallable
    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_out = self.conv2(conv1_out)
        if self.pre_output is not None:
            out = self.pre_output(conv2_out)
        else:
            out = conv2_out
        return out


class UNetSkeleton(nn.Module):
    def __init__(self, encoders, base, decoders, output, final_activation=None):
        super(UNetSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        # why do we hard-code this to 3 ? wouldn't it be enough to check that they are
        # all of the same length
        assert len(encoders) == len(decoders) == 3
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

        # all spatial sizes (ssize) for 512 input size
        # and donwscaling by a factor of 2
        # apply first decoder
        # e0.ssize = 256 (ssize / scale_factor)
        e0 = self.encoders[0](input_)

        # apply second encoder
        # e1.ssize = 128 (ssize / (scale_factor**2))
        e1 = self.encoders[1](e0)

        # apply third encoder
        # e2.ssize = 64 (ssize / (scale_factor**3))
        e2 = self.encoders[2](e1)

        # apply the base
        # b.ssize = 64
        b = self.base(e2)

        # apply the third / lowest decoder with input from base
        # and encoder 2
        # d2.ssize = 128
        d2 = self.decoders[0](
            torch.cat((b, e2), 1)
        )

        # apply the second decoder with input from the third decoder
        # and the second encoder
        # d1.ssize = 256
        d1 = self.decoders[1](
            torch.cat((d2, e1), 1)
        )

        # apply the first decoder with input from the second decoder
        # and the first encoder
        # d0.ssize = 512
        d0 = self.decoders[2](
            torch.cat((d1, e0), 1)
        )

        # out.ssize = 512
        out = self.output(d0)

        if self.final_activation is not None:
            out = self.final_activation(out)
        return out



class UNetSkeletonMultiscale(nn.Module):
    def __init__(self, encoders, base, decoders, predictors, final_activation=None):
        super(UNetSkeletonMultiscale, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        # why do we hard-code this to 3 ? wouldn't it be enough to check that they are
        # all of the same length
        assert len(encoders) == len(decoders) == 3
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.predictors = nn.ModuleList(predictors)

        self.base = base
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

    def apply_act(self, input_):
        if self.final_activation is not None:
            return self.final_activation(input_)
        else:
            return input_

    def forward(self, input_):

        # all spatial sizes (ssize) for 512 input size
        # and donwscaling by a factor of 2
        # apply first decoder
        # e0.ssize = 256 (ssize / scale_factor)
        e0 = self.encoders[0](input_)

        # apply second encoder
        # e1.ssize = 128 (ssize / (scale_factor**2))
        e1 = self.encoders[1](e0)

        # apply third encoder
        # e2.ssize = 64 (ssize / (scale_factor**3))
        e2 = self.encoders[2](e1)

        # apply the base
        # b.ssize = 64
        b = self.base(e2)

        # apply the third / lowest decoder with input from base
        # and encoder 2
        # d2.ssize = 128
        p3 = self.predictors[3](b)
        out3 = self.apply_act(p3)

        d2 = self.decoders[0](
            torch.cat((b, e2, p3), 1)
        )
        # apply the second decoder with input from the third decoder
        # and the second encoder
        # d1.ssize = 256
        p2 = self.predictors[2](d2)
        out2 = self.apply_act(p2)
        d1 = self.decoders[1](
            torch.cat((d2, e1, p2), 1)
        )
        # apply the first decoder with input from the second decoder
        # and the first encoder
        # d0.ssize = 512
        p1 = self.predictors[1](d1)
        out1 = self.apply_act(p1)
        d0 = self.decoders[2](
            torch.cat((d1, e0, p1), 1)
        )

        # out.ssize = 512
        p0 = self.predictors[0](d0)
        out0 = self.apply_act(p0)

        return out0, out1, out2, out3



