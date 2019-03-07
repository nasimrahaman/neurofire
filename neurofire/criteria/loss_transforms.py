import numbers
import warnings

import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d, conv3d

from inferno.io.transform import Transform

# TODO provide functionality to do trafos on gpu ?!?
# (for affinity trafos on gpu)


# TODO expect retain segmentation
class MaskIgnoreLabel(Transform):
    """
    """
    def __init__(self, ignore_label=0, **super_kwargs):
        super(MaskIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label

    # for all batch requests, we assume that
    # we are passed prediction and target in `tensors`
    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        mask_variable = Variable(target.data.clone().ne(float(self.ignore_label)).float(),
                                 requires_grad=False).expand_as(prediction)
        masked_prediction = prediction * mask_variable
        return masked_prediction, target


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


class MaskTransitionToIgnoreLabel(Transform):
    """Applies a mask where the transition to zero label is masked for the respective offsets."""
    def __init__(self, offsets, ignore_label=0,
                 mode='apply_mask_to_batch',
                 targets_are_inverted=True,
                 **super_kwargs):
        """
        Added additional parameter.
        :param mode: the default is 'apply_mask_to_batch'. An additional mode is 'return_mask'.
        """
        super(MaskTransitionToIgnoreLabel, self).__init__(**super_kwargs)
        assert isinstance(offsets, (list, tuple))
        assert len(offsets) > 0
        self.dim = len(offsets[0])
        self.offsets = offsets
        self.targets_are_inverted = targets_are_inverted
        assert isinstance(ignore_label, numbers.Integral)
        self.ignore_label = ignore_label

        assert mode == 'apply_mask_to_batch' or mode == 'return_mask', "Mode not recognized"
        self.mode = mode

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
        # FIXME: volatile is ignored starting from pytorch 0.4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dont_ignore_labels_mask_variable = Variable(segmentation.data.clone().ne_(self.ignore_label),
                                                    requires_grad=False, volatile=True)

        if self.dim == 2:
            kernel_alloc = segmentation.data.new(1, 1, 3, 3).zero_()
            conv = conv2d
        elif self.dim == 3:
            kernel_alloc = segmentation.data.new(1, 1, 3, 3, 3).zero_()
            conv = conv3d
        else:
            raise NotImplementedError

        shift_kernels = self.mask_shift_kernels(kernel_alloc, self.dim, offset)
        shift_kernels = Variable(shift_kernels, requires_grad=False)
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
        with torch.no_grad():
            # get the individual mask for the offsets
            masks = [self.mask_tensor_for_offset(segmentation, offset) for offset in self.offsets]
            # Concatenate to one tensor and convert tensor to variable
            out = torch.cat(tuple(masks), 1)
        return out

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        # validate the prediction
        assert prediction.dim() in [4, 5], prediction.dim()
        assert prediction.size(1) == len(self.offsets), "%i, %i" % (prediction.size(1), len(self.offsets))

        # validate target and extract segmentation from the target
        assert target.size(1) == len(self.offsets) + 1, "%i, %i" % (target.size(1), len(self.offsets) + 1)
        segmentation = target[:, 0:1]
        full_mask_variable = Variable(self.full_mask_tensor(segmentation), requires_grad=False)

        if self.mode == 'apply_mask_to_batch':
            # FIXME: should we not apply the mask also to the targets...?
            # Mask prediction with master mask
            masked_prediction = prediction * full_mask_variable
            targ_affinities = target[:, 1:]
            # Legacy:
            self.targets_are_inverted = self.targets_are_inverted if hasattr(self, 'targets_are_inverted') else True

            if self.targets_are_inverted:
                targ_affinities = 1 - targ_affinities
                targ_affinities = targ_affinities * full_mask_variable
                targ_affinities = 1 - targ_affinities
            else:
                targ_affinities = targ_affinities * full_mask_variable
            target = torch.cat((segmentation, targ_affinities), dim=1)
            return masked_prediction, target
        elif self.mode == 'return_mask':
            return full_mask_variable


class InvertTarget(Transform):
    def __init__(self, **super_kwargs):
        super(InvertTarget, self).__init__(**super_kwargs)

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors
        return prediction, 1. - target
