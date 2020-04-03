import numpy as np
import torch

from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import label

from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
import inferno.utils.torch_utils as tu

import logging
logger = logging.getLogger(__name__)


class DtypeMapping(object):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}
    INVERSE_DTYPE_MAPPING = {'float32': 'float',
                             'float64': 'double',
                             'float16': 'half',
                             'int64': 'long'}


# TODO rename to Segmentation2Edges ?!
# TODO implement retain segmentation
# TODO test for torch and np
class Segmentation2Membranes(Transform, DtypeMapping):
    """Convert dense segmentation to boundary-maps (or membranes)."""
    def __init__(self, dtype='float32', **super_kwargs):
        super(Segmentation2Membranes, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def image_function(self, image):
        if isinstance(image, np.ndarray):
            return self._apply_numpy_tensor(image)
        elif torch.is_tensor(image):
            return self._apply_torch_tensor(image)
        else:
            raise NotImplementedError("Only support np.ndarray and torch.tensor, got %s" % type(image))

    def _apply_numpy_tensor(self, image):
        gx = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(1, 3))
        gy = convolve(np.float32(image), np.array([-1., 0., 1.]).reshape(3, 1))
        return getattr(np, self.dtype)((gx ** 2 + gy ** 2) > 0)

    # TODO implement and test
    # def _apply_torch_tensor(self, image):
    #     conv = torch.nn.functional.conv2d
    #     kernel = image.new(1, 3, 3).zero_()
    #     return


class NegativeExponentialDistanceTransform(Transform):
    """'Smooth' e.g. membranes by applying a negative exponential on the distance transform."""
    def __init__(self, gain=1., invert=True, **super_kwargs):
        super(NegativeExponentialDistanceTransform, self).__init__(**super_kwargs)
        self.invert = invert
        self.gain = gain

    def image_function(self, image):
        if self.invert:
            image = 1. - image
            return np.exp(-self.gain * distance_transform_edt(image))
        else:
            # for ISBI the labels are inverted
            return 1 - np.exp(-self.gain * distance_transform_edt(image))


class ConnectedComponents2D(Transform):
    """
    Apply connected components on segmentation in 2D.
    """
    def __init__(self, **super_kwargs):
        """
        Parameters
        ----------
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents2D, self).__init__(**super_kwargs)

    def image_function(self, image):
        return label(image)


class ConnectedComponents3D(Transform):
    """
    Apply connected components on segmentation in 3D.
    """
    def __init__(self, **super_kwargs):
        """
        Parameters
        ----------
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents3D, self).__init__(**super_kwargs)

    def volume_function(self, volume):
        return label(volume)


# TODO refactor affogato functionality to public repo and make this obsolete
class Segmentation2AffinitiesFromOffsets(Transform, DtypeMapping):
    """ Fallback implementation for affinities if you can't use transforms
    defined in 'affinities.py'
    """

    def __init__(self, offsets, dtype='float32',
                 add_singleton_channel_dimension=False,
                 retain_segmentation=False, **super_kwargs):
        super(Segmentation2AffinitiesFromOffsets, self).__init__(**super_kwargs)
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        assert len(offsets) > 0, "`offsets` must not be empty."

        dim = len(offsets[0])
        assert dim in (2, 3), "Affinities are only supported for 2d and 3d input"
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        self.offsets = offsets if isinstance(offsets, int) else tuple(offsets)
        self.retain_segmentation = retain_segmentation

    def convolve_with_shift_kernel(self, tensor, offset):
        if isinstance(tensor, np.ndarray):
            return self._convolve_with_shift_kernel_numpy(tensor, offset)
        elif torch.is_tensor(tensor):
            return self._convolve_with_shift_kernel_torch(tensor, offset)
        else:
            raise NotImplementedError

    def build_shift_kernels(self, offset):
        if self.dim == 3:
            # Again, the kernels are similar to conv kernels in torch.
            # We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(1, 1, 3, 3, 3), dtype=self.dtype)

            assert len(offset) == 3
            assert np.sum(np.abs(offset)) > 0

            shift_combined[0, 0, 1, 1, 1] = -1.
            s_z = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            s_x = 1 if offset[2] == 0 else (2 if offset[2] > 0 else 0)
            shift_combined[0, 0, s_z, s_y, s_x] = 1.
            return shift_combined

        elif self.dim == 2:
            # Again, the kernels are similar to conv kernels in torch.
            # We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(1, 1, 3, 3), dtype=self.dtype)

            assert len(offset) == 2
            assert np.sum(np.abs(offset)) > 0

            shift_combined[0, 0, 1, 1] = -1.
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            shift_combined[0, 0, s_x, s_y] = 1.
            return shift_combined
        else:
            raise NotImplementedError

    def _convolve_with_shift_kernel_torch(self, tensor, offset):
        if self.dim == 3:
            # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
            # being channel
            assert tensor.dim() == 4, "Tensor must be 4D for dim = 3."
            assert tensor.size(0) == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv3d
        elif self.dim == 2:
            # Make sure the tensor contains 2D images (i.e. is 3D) with the first axis
            # being channel
            assert tensor.dim() == 3, "Tensor must be 3D for dim = 2."
            assert tensor.size(0) == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv2d
        else:
            raise NotImplementedError
        # Cast tensor to the right datatype (no-op if it's the right dtype already)
        tensor = getattr(tensor, self.INVERSE_DTYPE_MAPPING.get(self.dtype))()
        shift_kernel = torch.from_numpy(self.build_shift_kernels(offset))
        # Build torch variables of the right shape (i.e. with a leading singleton batch axis)
        torch_tensor = torch.autograd.Variable(tensor[None, ...])
        torch_kernel = torch.autograd.Variable(shift_kernel)
        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=abs_offset,
                               dilation=abs_offset)
        # Get rid of the singleton batch dimension (keep cuda tensor as is)
        convolved = torch_convolved.data[0, ...]
        return convolved

    def _convolve_with_shift_kernel_numpy(self, tensor, offset):
        if self.dim == 3:
            # Make sure the tensor is contains 3D volumes (i.e. is 4D) with the first axis
            # being channel
            assert tensor.ndim == 4, "Tensor must be 4D for dim = 3."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv3d
        elif self.dim == 2:
            # Make sure the tensor contains 2D images (i.e. is 3D) with the first axis
            # being channel
            assert tensor.ndim == 3, "Tensor must be 3D for dim = 2."
            assert tensor.shape[0] == 1, "Tensor must have only one channel."
            conv = torch.nn.functional.conv2d
        else:
            raise NotImplementedError
        # Cast tensor to the right datatype
        if tensor.dtype != self.dtype:
            tensor = tensor.astype(self.dtype)
        # Build torch variables of the right shape (i.e. with a leading singleton batch axis)
        torch_tensor = torch.autograd.Variable(torch.from_numpy(tensor[None, ...]))
        shift_kernel = self.build_shift_kernels(offset)
        torch_kernel = torch.autograd.Variable(torch.from_numpy(shift_kernel))

        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        # abs_offset = int(max(1, np.max(np.abs(offset))))
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=abs_offset,
                               dilation=abs_offset)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.cpu().numpy()[0, ...]
        return convolved

    def tensor_function(self, tensor):
        if isinstance(tensor, np.ndarray):
            return self._tensor_function_numpy(tensor)
        elif torch.is_tensor(tensor):
            return self._tensor_function_torch(tensor)
        else:
            raise NotImplementedError("Only support np.ndarray and torch.tensor, got %s" % type(tensor))

    def _tensor_function_torch(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.dim() not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.dim() == 3 and self.dim == 2) or (tensor.dim() == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = torch.cat([self.convolve_with_shift_kernel(tensor, offset)
                                         for offset in self.offsets], dim=0)
        elif tensor.dim() == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            assert False, "Not implemented yet"
            convolved_tensor = torch.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                            for z_num in range(tensor.size(1))], dim=1)
        else:
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = tu.where(convolved_tensor == 0.,
                                        convolved_tensor.new(*convolved_tensor.size()).fill_(1.),
                                        convolved_tensor.new(*convolved_tensor.size()).fill_(0.))

        # We might want to carry the segmentation along (e.g. when combining MALIS with
        # euclidean loss higher-order affinities). If this is the case, we insert the segmentation
        # as the *first* channel.
        if self.retain_segmentation:
            tensor = getattr(tensor, self.INVERSE_DTYPE_MAPPING.get(self.dtype))()
            output = torch.cat((tensor, binarized_affinities), 0)
        else:
            output = binarized_affinities
        return output

    def _tensor_function_numpy(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.ndim not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.ndim == 3 and self.dim == 2) or (tensor.ndim == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = np.concatenate(
                [self.convolve_with_shift_kernel(tensor, offset)
                 for offset in self.offsets], axis=0)
        elif tensor.ndim == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            # TODO
            assert False, "Not implemented yet"
            convolved_tensor = np.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                         for z_num in range(tensor.shape[1])], axis=1)
        else:
            print(tensor.ndim, self.dim)
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = np.where(convolved_tensor == 0., 1., 0.)
        # Cast to be sure
        if not binarized_affinities.dtype == self.dtype:
            binarized_affinities = binarized_affinities.astype(self.dtype)

        if self.retain_segmentation:
            if tensor.dtype != self.dtype:
                tensor = tensor.astype(self.dtype)
            output = np.concatenate((tensor, binarized_affinities), axis=0)
        else:
            output = binarized_affinities
        return output
