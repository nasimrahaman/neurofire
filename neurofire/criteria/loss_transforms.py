import numbers

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d, conv3d

from inferno.io.transform import Transform


class DropChannels(Transform):
    """
    """
    def __init__(self, index, from_, **super_kwargs):
        super().__init__(**super_kwargs)
        assert isinstance(index, (int, list, tuple)),\
            "Only supports channels specified by single number or list / tuple"
        self.index = index if isinstance(index, (list, tuple)) else [index]
        assert all(ind >= 0 for ind in self.index)

        if from_ == 'prediction':
            self.drop_in_prediction = True
            self.drop_in_target = False
        elif from_ == 'target':
            self.drop_in_prediction = False
            self.drop_in_target = True
        elif from_ == 'both':
            self.drop_in_prediction = True
            self.drop_in_target = True
        else:
            raise ValueError("%s option for parameter `from_` not supported" % from_)

    def _drop_channels(self, tensor):
        n_channels = tensor.shape[1]
        assert all(index < n_channels for index in self.index), "%s, %s" % (str(self.index), str(n_channels))
        keep_axis = [index for index in range(n_channels) if index not in self.index]
        return tensor[:, keep_axis]

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        if self.drop_in_prediction:
            prediction = self._drop_channels(prediction)
        if self.drop_in_target:
            target = self._drop_channels(target)
        return prediction, target


class ExpPrediction(Transform):
    """
    """
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return torch.exp(prediction), target


class SoftmaxPrediction(Transform):
    """
    """
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return F.softmax(prediction, dim=1), target


class SigmoidPrediction(Transform):
    """
    """
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return torch.sigmoid(prediction), target


class OrdinalToOneHot(Transform):
    """
    """
    def __init__(self, n_classes, ignore_label=None, **super_kwargs):
        super().__init__(**super_kwargs)
        if isinstance(n_classes, int):
            self.n_classes = n_classes
            self.classes = list(range(n_classes))
            self.class_to_channel = None
        elif isinstance(n_classes, (list, tuple)):
            self.n_classes = len(n_classes)
            self.classes = n_classes
            self.class_to_channel = {class_id: chan_id
                                     for chan_id, class_id in enumerate(self.classes)}
        else:
            raise ValueError("Unsupported type %s" % type(n_classes))
        self.ignore_label = ignore_label

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        assert prediction.shape[1] == self.n_classes
        transformed = torch.zeros_like(prediction)
        for c in self.classes:
            chan = c if self.class_to_channel is None else self.class_to_channel[c]
            transformed[:, chan:chan+1] += target.eq(float(c)).float()
        # preserve the ignore-label
        if self.ignore_label is not None:
            mask = target.eq(float(self.ignore_label))
            for c in range(self.n_classes):
                transformed[:, c:c+1][mask] = self.ignore_label
        return prediction, transformed


class SqueezeSingletonAxis(Transform):
    """
    """
    def __init__(self, axis=0, **super_kwargs):
        super().__init__(**super_kwargs)
        self.axis = axis

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        shape = target.shape
        assert shape[self.axis] == 1
        slice_ = tuple(slice(None) if dim != self.axis else 0
                       for dim in range(len(shape)))
        target = target[slice_]
        return prediction, target


class MaskIgnoreLabel(Transform):
    """
    """
    def __init__(self, ignore_label=0, set_to_zero=False, **super_kwargs):
        super(MaskIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label
        self.set_to_zero = set_to_zero

    # for all batch requests, we assume that
    # we are passed prediction and target in `tensors`
    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        mask = target.clone().ne_(float(self.ignore_label))
        if self.set_to_zero:
            target[torch.eq(mask, 0)] = 0
        mask_tensor = mask.float().expand_as(prediction)
        mask_tensor.requires_grad = False
        masked_prediction = prediction * mask_tensor
        return masked_prediction, target


# TODO remove this in favor of more general `DropChannels`
class RemoveSegmentationFromTarget(Transform):
    """
    Remove the zeroth channel (== segmentation when `retain_segmentation` is used)
    from the target.
    """
    def __init__(self, **super_kwargs):
        super(RemoveSegmentationFromTarget, self).__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return prediction, target[:, 1:]


class ApplyAndRemoveMask(Transform):
    def __init__(self, **super_kwargs):
        super(ApplyAndRemoveMask, self).__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors

        # validate the prediction
        assert prediction.dim() in [4, 5], prediction.dim()
        assert target.dim() == prediction.dim(), "%i, %i" % (target.dim(), prediction.dim())
        assert target.size(1) == 2 * prediction.size(1), "%i, %i" % (target.size(1), prediction.size(1))
        assert target.shape[2:] == prediction.shape[2:], "%s, %s" % (str(target.shape), str(prediction.shape))
        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        mask.requires_grad = False

        # mask prediction with mask
        masked_prediction = prediction * mask
        return masked_prediction, target


class MaskTransitionToIgnoreLabel(Transform):
    """Applies a mask where the transition to zero label is masked for the respective offsets."""

    def __init__(self, offsets, ignore_label=0,
                 skip_channels=None, **super_kwargs):
        super(MaskTransitionToIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(offsets, (list, tuple))
        assert len(offsets) > 0
        self.dim = len(offsets[0])
        self.offsets = offsets
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label
        self.skip_channels = skip_channels

    # TODO explain what the hell is going on here ...
    @staticmethod
    def mask_shift_kernels(kernel, dim, offset):
        if dim == 3:
            assert len(offset) == 3
            s_z = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            s_x = 1 if offset[2] == 0 else (2 if offset[2] > 0 else 0)
            kernel[0, 0, s_z, s_y, s_x] = 1.
        elif dim == 2:
            assert len(offset) == 2
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            kernel[0, 0, s_x, s_y] = 1.
        else:
            raise NotImplementedError
        return kernel

    def get_dont_ignore_labels_mask(self, segmentation, offset):
        return segmentation.data.clone().ne_(self.ignore_label)

    def mask_tensor_for_offset(self, segmentation, offset):
        """
        Generate mask where a pixel is 1 if it's NOT a transition to ignore label
        AND not a ignore label itself.

        Example (ignore label 0)
        -------
        For
            offset       = 2,
            segmentation = 0 0 0 1 1 1 1 2 2 2 2 0 0 0
            affinity     = 0 1 1 1 1 0 0 1 1 0 0 1 1 0
            shift_mask   = 0 0 0 0 0 1 1 1 1 1 1 1 1 0
        --> final_mask   = 0 0 0 0 0 1 1 1 1 1 1 0 0 0
        """
        # expecting target to be segmentation of shape (N, 1, z, y, x)
        assert segmentation.size(1) == 1, str(segmentation.size())

        # Get mask where we don't have ignore label
        dont_ignore_labels_mask_variable = self.get_dont_ignore_labels_mask(segmentation, offset)
        dont_ignore_labels_mask_variable.requires_grad = False

        if self.dim == 2:
            kernel_alloc = segmentation.data.new(1, 1, 3, 3).zero_()
            conv = conv2d
        elif self.dim == 3:
            kernel_alloc = segmentation.data.new(1, 1, 3, 3, 3).zero_()
            conv = conv3d
        else:
            raise NotImplementedError

        shift_kernels = self.mask_shift_kernels(kernel_alloc, self.dim, offset)
        shift_kernels.requires_grad = False
        # Convolve
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        mask_shifted = conv(input=dont_ignore_labels_mask_variable,
                            weight=shift_kernels,
                            padding=abs_offset, dilation=abs_offset)
        # Mask the mask tehe
        final_mask_tensor = (dont_ignore_labels_mask_variable
                             .expand_as(mask_shifted)
                             .data
                             .mul_(mask_shifted.data))
        return final_mask_tensor

    # get the full mask tensor
    def full_mask_tensor(self, segmentation):
        # get the individual mask for the offsets
        masks = [self.mask_tensor_for_offset(segmentation, offset) for offset in self.offsets]
        # Concatenate to one tensor and convert tensor to variable
        return torch.cat(tuple(masks), 1)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # validate the prediction
        assert prediction.dim() in [4, 5], prediction.dim()
        assert prediction.size(1) == len(self.offsets), "%i, %i" % (prediction.size(1), len(self.offsets))

        # validate target and extract segmentation from the target
        assert target.size(1) == len(self.offsets) + 1, "%i, %i" % (target.size(1), len(self.offsets) + 1)
        segmentation = target[:, 0:1]
        full_mask_variable = self.full_mask_tensor(segmentation)
        full_mask_variable.requires_grad = False

        # Mask prediction with master mask
        masked_prediction = prediction * full_mask_variable
        return masked_prediction, target


class InvertTarget(Transform):
    def __init__(self, **super_kwargs):
        super(InvertTarget, self).__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return prediction, 1. - target


class InvertPrediction(Transform):
    def __init__(self, **super_kwargs):
        super(InvertPrediction, self).__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return 1. - prediction, target


class RemoveIgnoreLabel(Transform):
    def __init__(self, ignore_label=0, **super_kwargs):
        super(RemoveIgnoreLabel, self).__init__(**super_kwargs)
        assert ignore_label == 0, "Only ignore label 0 is supported so far"
        self.ignore_label = ignore_label

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # for now, we just increase the target by 1
        # in the general case, we should check if we have the ignore label
        # and then replace it
        target += 1
        return prediction, target


class AddNoise(Transform):
    """ Add noise to the inputs before applying loss
    """
    def __init__(self, apply_='prediction', noise_type='uniform', **noise_kwargs):
        super(AddNoise, self).__init__()
        assert apply_ in ('target', 'prediction', 'both'), apply_
        self.apply_ = apply_
        assert noise_type in ('gaussian', 'uniform', 'gumbel'), noise_type
        if noise_type == 'gaussian':
            self._noise = self._gaussian_noise
            self.mean = noise_kwargs.get('mean', 0)
            self.std = noise_kwargs.get('std', 1)
        elif noise_type == 'uniform':
            self._noise = self._uniform_noise
            self.min = noise_kwargs.get('min', 0)
            self.max = noise_kwargs.get('max', 1)
        elif noise_type == 'gumbel':
            self._noise = self._gumbel_noise
            self.loc = noise_kwargs.get('loc', 0)
            self.scale = noise_kwargs.get('scale', 1)

    @staticmethod
    def _scale(input_, min_, max_):
        input_ = (input_ - input_.min())
        input_ = input_ / input_.max()
        input_ = (max_ - min_) * input_ + min_
        return input_

    def _uniform_noise(self, input_,):
        imin, imax = input_.min(), input_.max()
        # TODO do we need the new empty ???
        noise = input_.new_empty(size=input_.size()).uniform_(self.min, self.max)
        input_ = input_ + noise
        input_ = self._scale(input_, imin, imax)
        return input_

    def _gaussian_noise(self, input_,):
        imin, imax = input_.min(), input_.max()
        # TODO do we need the new empty ???
        noise = input_.new_empty(size=input_.size()).normal_(self.mean, self.std)
        input_ = input_ + noise
        input_ = self._scale(input_, imin, imax)
        return input_

    def _gumbel_noise(self, input_,):
        imin, imax = input_.min(), input_.max()
        noise = np.random.gumbel(loc=self.loc, scale=self.scale,
                                 size=input_.shape)
        input_ = input_ + torch.from_numpy(noise)
        input_ = self._scale(input_, imin, imax)
        return input_

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        if self.apply_ in ('prediction', 'both'):
            prediction = self._noise(prediction)
        if self.apply_ in ('target', 'both'):
            target = self._noise(target)
        return prediction, target


class AlignPredictionAndTarget(Transform):
    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        pshape = prediction.shape[2:]
        tshape = target.shape[2:]

        crop_target = any(tsh > psh for tsh, psh in zip(tshape, pshape))
        crop_pred = any(psh > tsh for tsh, psh in zip(tshape, pshape))
        if crop_target and crop_pred:
            raise RuntimeError("Inconsistent target and prediction sizes")

        if crop_target:
            shape_diff = [(tsh - psh) // 2 for tsh, psh in zip(tshape, pshape)]
            bb = tuple(slice(sd, tsh - sd) for sd, tsh in zip(shape_diff, tshape))
            bb = np.s_[:, :] + bb
            target = target[bb]
        elif crop_pred:
            shape_diff = [(psh - tsh) // 2 for tsh, psh in zip(tshape, pshape)]
            bb = tuple(slice(sd, psh - sd) for sd, tsh in zip(shape_diff, tshape))
            bb = np.s_[:, :] + bb
            prediction = prediction[bb]

        return prediction, target
