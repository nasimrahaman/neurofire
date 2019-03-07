import torch
import torch.nn as nn


class WNetSkeleton(nn.Module):
    def __init__(self, encoders1, decoders1, base1,
                 encoders2, decoders2, base2, output,
                 final_activation=None):
        super(WNetSkeleton, self).__init__()
        assert isinstance(encoders1, list)
        assert isinstance(decoders1, list)
        assert isinstance(encoders2, list)
        assert isinstance(decoders2, list)
        assert isinstance(base1, nn.Module)
        assert isinstance(base2, nn.Module)

        assert len(encoders1) == len(encoders2) == len(decoders1) == len(decoders2)
        self.encoders1 = nn.ModuleList(encoders1)
        self.decoders1 = nn.ModuleList(decoders1)
        self.encoders2 = nn.ModuleList(encoders2)
        self.decoders2 = nn.ModuleList(decoders2)

        self.base1 = base1
        self.base2 = base2
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
        encoder_out1 = []
        # apply encoders of the first encoder path
        # and remember their outputs
        for encoder in self.encoders1:
            x = encoder(x)
            encoder_out1.append(x)

        # apply first base and remember its output
        x = self.base1(x)
        decoder_out1 = [x]

        # apply decoders of the first decoder path
        # and remember their outputs
        max_level = len(self.decoders1) - 1
        for level, decoder in enumerate(self.decoders1):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((x, encoder_out1[max_level - level]), 1))
            decoder_out1.append(x)
        # reverse decoder output
        decoder_out1 = decoder_out1[::-1]

        # apply the encoders of the second encoder path
        # they get additional input from the decoder path 1 and
        # their output is added to the output of the first encoders
        for level, encoder in enumerate(self.encoders2):
            if level == 0:
                x = encoder(x)
            else:
                x = encoder(torch.cat((x, decoder_out1[level]), 1))
            # add the output oup with the output from the previous encoder path
            encoder_out1[level] = encoder_out1[level] + x

        # apply the second base
        x = self.base2(torch.cat((x, decoder_out1[-1]), 1))

        # apply decoders of the second decoder path
        # and remember their outputs
        for level, decoder in enumerate(self.decoders2):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((x, encoder_out1[max_level - level]), 1))

        x = self.output(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
