import torch.nn as nn

from inferno.extensions.layers.sampling import AnisotropicPool
from .loss_wrapper import LossWrapper


class MultiScaleLoss(nn.Module):
    # Steffen's arguments `offsets` and `scale_facter` were never used
    def __init__(self, loss, n_scales=4, scale_weights=None, fill_missing_targets=False):
        super(MultiScaleLoss, self).__init__()
        assert isinstance(loss, (LossWrapper, nn.Module))
        self.loss = loss
        self.n_scales = n_scales
        # per default, we weight each scale's loss with 1 / 4**scale_level i. (1, 1/4, 1/16, 1/128, ...)
        if scale_weights is None:
            self.scale_weights = [1. / 4**scale for scale in range(n_scales)]
        else:
            assert isinstance(scale_weights, (list, tuple))
            assert len(scale_weights) == n_scales
            self.scale_weights = scale_weights
        # flag to indicate if we should fill up missing targets if
        # the length of prediction and target is not the same
        self.fill_missing_targets = fill_missing_targets

    def forward(self, predictions, targets):
        assert isinstance(predictions, (list, tuple)), type(predictions)
        assert len(predictions) == self.n_scales, "%i, %i" % (len(predictions), self.n_scales)
        same_len = len(predictions) == len(targets)

        # if we have different number of targets and predictions, we might have a fusion layer
        # that produces predictions at the original scale. In this case, we can just fill up the missing
        # targets with the 0-level target. (only if fill_missing_targets == True)
        if not same_len:
            assert self.fill_missing_targets and len(predictions) > len(targets), "%i, %i" % (len(predictions),
                                                                                              len(targets))
            n_missing = len(predictions) - len(targets)
            targets = n_missing * [targets[0]] + targets

        # TODO make sure that this actually checks out with pytorch logics
        # calculate and add up the loss for each scale, weighted by the corresponding factor
        # (weighting factor 0 disables the scale)
        loss = sum([self.loss(ps, ts) * ws
                    for ps, ts, ws in zip(predictions, targets, self.scale_weights) if ws > 0])
        return loss


# TODO
# - this should go somewhere else (inferno.extensions)
# - should check if some existing torch functionality can be used (interpolation nearest)
class Downsampler(object):
    def __init__(self, scale_factor, ndim=None):
        assert isinstance(scale_factor, (list, int, tuple))
        if isinstance(scale_factor, (list, tuple)):
            assert all(isinstance(sf, int) for sf in scale_factor)
            if ndim is None:
                self.ndim = len(scale_factor)
            else:
                assert len(scale_factor) == ndim
                self.ndim = ndim
            self.scale_factor = scale_factor
        else:
            assert ndim is not None, "Cannot infer dimension from scalar downsample factor"
            self.ndim = ndim
            self.scale_factor = self.ndim * (scale_factor,)
        self.ds_slice = tuple(slice(None, None, sf) for sf in scale_factor)

    def __call__(self, input_):
        if input_.ndim > self.ndim:
            assert input_.ndim == self.ndim + 1, "%i, %i" % (input_.ndim, self.ndim)
            ds_slice = (slice(None),) + self.ds_slice
        else:
            ds_slice = self.ds_slice
        return input_[ds_slice]


class MultiScaleLossMaxPool(MultiScaleLoss):
    # NOTE if invert target is activated here, it should not be added in the loss transforms
    # otherwise it will be applied twice
    def __init__(self, loss, scaling_factor, n_scales=4, scale_weights=None,
                 invert_target=True, retain_segmentation=False):
        super(MultiScaleLossMaxPool, self).__init__(loss, n_scales, scale_weights)
        assert isinstance(scaling_factor, (list, tuple, int))
        if isinstance(scaling_factor, int):
            self.scaling_factor = (self.n_scales - 1) * [scaling_factor]
        else:
            assert len(scaling_factor) == self.n_scales - 1
            self.scaling_factor = scaling_factor

        self.poolers = []
        for scale_factor in self.scaling_factor:
            if isinstance(scale_factor, (list, tuple)):
                assert len(scale_factor) == 3
                # we need to make sure that the scale factor conforms with the single value
                # that AnisotropicPool expects
                assert scale_factor[0] == 1
                assert scale_factor[1] == scale_factor[2]
                sampler = AnisotropicPool(downscale_factor=scale_factor[1])
            else:
                sampler = nn.MaxPool3d(kernel_size=1 + scale_factor,
                                       stride=scale_factor,
                                       padding=1)
            self.poolers.append(sampler)

        self.invert_target = invert_target
        self.retain_segmentation = retain_segmentation

        # if retain segmentation is activated,
        # we need to add transformations for the segmentations as well
        # TODO generalize this to 2 and 3D
        if self.retain_segmentation:
            self.samplers = [Downsampler(sf) for sf in self.scaling_factor]

    def forward(self, predictions, target):
        assert isinstance(predictions, (list, tuple))
        assert len(predictions) == self.n_scales

        if self.invert_target:
            if self.retain_segmentation:
                target[:, 1:] = 1. - target[:, 1:]
            else:
                target = 1. - target

        targets = [target]
        # if retain_segmentation is true, we need to pool
        # differently for channel 0 (= segmentation) and the affinty channels
        if self.retain_segmentation:
            for scale in range(self.n_scales - 1):
                segmentation = self.samplers[scale](target[:, :1])
                target = self.poolers[scale](target)
                target[:, 0] = segmentation
                targets.append(target)
        else:
            for scale in range(self.n_scales - 1):
                target = self.poolers[scale](target)
                targets.append(target)

        return super(MultiScaleLossMaxPool, self).forward(predictions, targets)
