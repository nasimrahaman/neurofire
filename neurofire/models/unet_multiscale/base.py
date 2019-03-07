import torch
import torch.nn as nn


class UNetSkeletonMultiscale(nn.Module):
    def __init__(self, encoders, base, decoders, predictors,
                 samplers, final_activation=None,
                 return_inner_feature_layers=False):
        super(UNetSkeletonMultiscale, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)
        assert isinstance(predictors, list)

        assert isinstance(return_inner_feature_layers, bool)
        if return_inner_feature_layers:
            raise NotImplementedError()
        self._return_inner_feature_layers = return_inner_feature_layers

        assert len(samplers) == len(encoders) == len(decoders) == len(predictors) - 1
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.samplers = nn.ModuleList(samplers)
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

        x = input_
        encoder_out = []
        # apply encoders and remember their outputs
        for encoder in self.encoders:
            x = encoder(x)
            encoder_out.append(x)

        x = self.base(x)
        outputs = [self.predictors[-1](x)]

        # apply decoders
        max_level = len(self.decoders) - 1
        for level, decoder in enumerate(self.decoders):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(torch.cat((self.samplers[level](x),
                                   encoder_out[max_level - level],
                                   self.samplers[level](outputs[-1])), 1))
            outputs.append(self.predictors[max_level - level](x))

        outputs = tuple(self.apply_act(out) for out in reversed(outputs))
        return outputs
