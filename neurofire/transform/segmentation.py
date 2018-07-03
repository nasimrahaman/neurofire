import numpy as np
import torch
from torch.autograd import Variable

from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label

from inferno.io.transform import Transform
import inferno.utils.python_utils as pyu
import inferno.utils.torch_utils as tu

from ..criteria.loss_transforms import MaskTransitionToIgnoreLabel

import logging
logger = logging.getLogger(__name__)

try:
    import vigra
    with_vigra = True
except ImportError:
    logger.warn("Vigra was not found, connected components will not be available")
    vigra = None
    with_vigra = False


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


# TODO remain to Segmentation2Edges ?!
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
    def _apply_torch_tensor(self, image):
        conv = torch.nn.functional.conv2d
        kernel = image.new(1, 3, 3).zero_()
        return


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


class Segmentation2Affinities(Transform, DtypeMapping):
    """Convert dense segmentation to affinity-maps of arbitrary order."""
    def __init__(self, dim, order=1, dtype='float32', add_singleton_channel_dimension=False,
                 retain_segmentation=False, **super_kwargs):
        super(Segmentation2Affinities, self).__init__(**super_kwargs)
        # Privates
        self._shift_kernels = None
        # Validate and register args
        assert dim in [2, 3]
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        self.order = order if isinstance(order, int) else tuple(order)
        self.retain_segmentation = retain_segmentation
        # Build kernels
        self._shift_kernels = self.build_shift_kernels(dim=self.dim, dtype=self.dtype)

    @staticmethod
    def build_shift_kernels(dim, dtype):
        if dim == 3:
            # The kernels have a shape similar to conv kernels in torch. We have 3 output channels,
            # corresponding to (depth, height, width)
            shift_combined = np.zeros(shape=(3, 1, 3, 3, 3), dtype=dtype)
            # Shift depth
            shift_combined[0, 0, 0, 1, 1] = 1.
            shift_combined[0, 0, 1, 1, 1] = -1.
            # Shift height
            shift_combined[1, 0, 1, 0, 1] = 1.
            shift_combined[1, 0, 1, 1, 1] = -1.
            # Shift width
            shift_combined[2, 0, 1, 1, 0] = 1.
            shift_combined[2, 0, 1, 1, 1] = -1.
            # Set
            return shift_combined
        elif dim == 2:
            # Again, the kernels are similar to conv kernels in torch. We now have 2 output
            # channels, corresponding to (height, width)
            shift_combined = np.zeros(shape=(2, 1, 3, 3), dtype=dtype)
            # Shift height
            shift_combined[0, 0, 0, 1] = 1.
            shift_combined[0, 0, 1, 1] = -1.
            # Shift width
            shift_combined[1, 0, 1, 0] = 1.
            shift_combined[1, 0, 1, 1] = -1.
            # Set
            return shift_combined
        else:
            raise NotImplementedError

    def convolve_with_shift_kernel(self, tensor):
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
        torch_kernel = torch.autograd.Variable(torch.from_numpy(self._shift_kernels))
        # Apply convolution (with zero padding). To obtain higher order features,
        # we apply a dilated convolution.
        torch_convolved = conv(input=torch_tensor,
                               weight=torch_kernel,
                               padding=self.order,
                               dilation=self.order)
        # Extract numpy array and get rid of the singleton batch dimension
        convolved = torch_convolved.data.numpy()[0, ...]
        return convolved

    def tensor_function(self, tensor):
        # Add singleton channel dimension if requested
        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]
        if tensor.ndim not in [3, 4]:
            raise NotImplementedError("Affinity map generation is only supported in 2D and 3D. "
                                      "Did you mean to set add_singleton_channel_dimension to "
                                      "True?")
        if (tensor.ndim == 3 and self.dim == 2) or (tensor.ndim == 4 and self.dim == 3):
            # Convolve tensor with a shift kernel
            convolved_tensor = self.convolve_with_shift_kernel(tensor)
        elif tensor.ndim == 4 and self.dim == 2:
            # Tensor contains 3D volumes, but the affinity maps are computed in 2D. So we loop over
            # all z-planes and concatenate the results together
            convolved_tensor = np.stack([self.convolve_with_shift_kernel(tensor[:, z_num, ...])
                                         for z_num in range(tensor.shape[1])], axis=1)
        else:
            raise NotImplementedError
        # Threshold convolved tensor
        binarized_affinities = np.where(convolved_tensor == 0., 1., 0.)
        # Cast to be sure
        if not binarized_affinities.dtype == self.dtype:
            binarized_affinities = binarized_affinities.astype(self.dtype)
        # We might want to carry the segmentation along (e.g. when combining MALIS with
        # euclidean loss higher-order affinities). If this is the case, we insert the segmentation
        # as the *first* channel.
        if self.retain_segmentation:
            if tensor.dtype != self.dtype:
                tensor = tensor.astype(self.dtype)
            output = np.concatenate((tensor, binarized_affinities), axis=0)
        else:
            output = binarized_affinities
        return output


def get_boundary_offsets(boundary_erode_segmentation):
    # print("WARNING: boundary erosion not properly working with ignore label")
    # print("WARNING: boundary erosion has artifacts at the image border")
    assert isinstance(boundary_erode_segmentation, list)
    dim = len(boundary_erode_segmentation)

    # Center the offset to grow a symmetric boundary:
    boundary_offsets = [[bound_offset if i == j else 0 for j in range(dim)] for i, bound_offset in
                        enumerate(boundary_erode_segmentation) if bound_offset != 0]
    negative_boundary_offsets = [[-int(val / 2) for val in off] for off in boundary_offsets]
    boundary_offsets = [[int(val / 2) if val % 2 == 0 else int(val / 2) + 1 for val in off] for off in boundary_offsets]

    # First we compute the boundary:
    return boundary_offsets + [off for off in negative_boundary_offsets if np.array(off).sum() != 0]


class Segmentation2AffinitiesFromOffsets(Transform, DtypeMapping):
    def __init__(self, dim, offsets, dtype='float32',
                 add_singleton_channel_dimension=False,
                 use_gpu=False,
                 retain_segmentation=False,
                 boundary_erode_segmentation=None,
                 return_eroded_labels=False,
                 ignore_label=0,
                 **super_kwargs):
        """
        :param boundary_erode_segmentation: If not None, it is a list with len==dim specifying the thickness
        of the erosion in every dimension.
        """
        super(Segmentation2AffinitiesFromOffsets, self).__init__(**super_kwargs)
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        assert len(offsets) > 0, "`offsets` must not be empty."
        assert ignore_label >= 0

        assert dim in (2, 3), "Affinities are only supported for 2d and 3d input"
        self.dim = dim
        self.dtype = self.DTYPE_MAPPING.get(dtype)
        self.add_singleton_channel_dimension = bool(add_singleton_channel_dimension)
        self.offsets = offsets if isinstance(offsets, int) else tuple(offsets)
        self.retain_segmentation = retain_segmentation
        self.use_gpu = use_gpu

        self.erode_boundary = True if boundary_erode_segmentation is not None else False
        self.return_eroded_labels = return_eroded_labels
        if boundary_erode_segmentation is not None:
            assert len(boundary_erode_segmentation) == dim
            boundary_offsets = get_boundary_offsets(boundary_erode_segmentation)

            self.compute_erosion_boundary = Segmentation2AffinitiesFromOffsets(dim, boundary_offsets,
                                               dtype=dtype,
                                               use_gpu=use_gpu,
                                               retain_segmentation=False,
                                               add_singleton_channel_dimension=add_singleton_channel_dimension,
                                               boundary_erode_segmentation=None,
                                               **super_kwargs)

            self.mask_ignore_label = MaskTransitionToIgnoreLabel(boundary_offsets, ignore_label=ignore_label, mode='return_mask',
                                                                     targets_are_inverted=True)

            # Then we mask any transition to the boundary:
            self.mask_erosion_boundary = MaskTransitionToIgnoreLabel(offsets, ignore_label=0,mode='return_mask',
                                        targets_are_inverted=True)
        self.boundary_erode_segmentation = boundary_erode_segmentation

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
        # Move tensor to GPU if required
        if self.use_gpu:
            tensor = tensor.cuda()
            shift_kernel = shift_kernel.cuda()
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

        # Move tensor to GPU if required
        if self.use_gpu:
            torch_tensor = torch_tensor.cuda()
            torch_kernel = torch_kernel.cuda()
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
            output = self._tensor_function_numpy(tensor)
        elif torch.is_tensor(tensor):
            output =  self._tensor_function_torch(tensor)
        else:
            raise NotImplementedError("Only support np.ndarray and torch.tensor, got %s" % type(tensor))

        if self.erode_boundary:
            assert torch.is_tensor(tensor), "Numpy boundary erosion not implemented yet"
            output = self._compute_boundary_erosion(tensor, output)

        return output

    def _compute_boundary_erosion(self, tensor, output):
        # FIXME: problem with ignore label... There will be part of the boundary with ignore_label now that will not be masked...!
        # FIXME: a boundary is detected also at the border of the image!

        # First we compute the boundary mask: (should be zero when there is a boundary)
        boundary_affs = self.compute_erosion_boundary.tensor_function(tensor).byte()



        if self.add_singleton_channel_dimension:
            tensor = tensor[None, ...]

        # The mask should be zero when an ignore label is involved:
        ignore_mask = self.mask_ignore_label.full_mask_tensor(
            Variable(tensor[0].float()[None, None, ...], requires_grad=False))[0].byte()


        boundary_mask = torch.ones(tensor[0].size()).byte()
        if boundary_affs.is_cuda:
            boundary_mask = boundary_mask.cuda()
            ignore_mask = ignore_mask.cuda()
        for i in range(boundary_affs.size()[0]):
            boundary_mask = boundary_mask * (1 - (boundary_affs[i] == 0) * (ignore_mask[i] == 1))



        segmentation_labels = tensor[0].float() + 1
        segmentation_labels = segmentation_labels * boundary_mask.float()

        # Then we set to 'active/split' the affinities that involves the boundary:
        # The mask is zero when a boundary is involved:
        affs_mask = self.mask_erosion_boundary.full_mask_tensor(Variable(segmentation_labels[None, None, ...], requires_grad=False))[0]
        if self.retain_segmentation:
            output[1:] = output[1:] * affs_mask
            if self.return_eroded_labels:
                output[0] = segmentation_labels
        else:
            output = output * affs_mask




        return output

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

        if self.erode_boundary:
            pass

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

        if self.erode_boundary:
            raise NotImplementedError()
        return output


def shift_tensor(tensor, random_offset):
    padding = [(0,0)]
    slicing = [slice(None)]
    for o in random_offset:
        padding.append((max(0,-o), max(0, o)))
        if o == 0:
            slicing.append(slice(None))
        elif o > 0:
            slicing.append(slice(o, None))
        else:
            slicing.append(slice(None, o))

    shifted_tensor = np.pad(tensor, padding, "constant")
    tmp = shifted_tensor[slicing]
    return np.concatenate((tensor, shifted_tensor[slicing]), 0)


class ShiftImageAndSegmentationAffinitiesWithRandomOffsets(Transform):
    """
    This transformation applies a random offset shift to the complete training batch
    the input image is shifted by the random offset and concatenated as an additional channel
    the label is transformed to the corresponding affinities using Segmentation2AffinitiesFromOffsets
    """
    def __init__(self, dim, offset_range, dtype='float32',
                 use_gpu=False,
                 **super_kwargs):
        self.dim = dim
        self.dtype = dtype
        self.offset_range = offset_range
        self.use_gpu = use_gpu
        super(ShiftImageAndSegmentationAffinitiesWithRandomOffsets, self).__init__(**super_kwargs)

    def get_offset(self):
        ora = self.offset_range
        np.random.seed()
        random_offset = [np.random.randint(-ora[i], ora[i])
                            if ora[i] > 0 else 0 for i in range(self.dim)]
        if all(ro == 0 for ro in random_offset):
            # draw again if all offsets are zero
            return self.get_offset()
        else:
            return random_offset

    def batch_function(self, tensors):
        random_offset = self.get_offset()
        s2a = Segmentation2AffinitiesFromOffsets(self.dim, [random_offset], dtype='float32',
                 add_singleton_channel_dimension=True,
                 use_gpu=self.use_gpu,
                 retain_segmentation=True
            )
        return shift_tensor(tensors[0], random_offset), s2a.tensor_function(tensors[1])


class ConnectedComponents2D(Transform):
    """
    Apply connected components on segmentation in 2D.
    """
    def __init__(self, label_segmentation=True, **super_kwargs):
        """
        Parameters
        ----------
        label_segmentation : bool
            Whether the input is a segmentation. If True (default), this computes a
            connected components on both segmentation and binary images (instead of just binary
            images, when this is set to False). However, this would require vigra as a dependency.
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents2D, self).__init__(**super_kwargs)
        self.label_segmentation = label_segmentation

    def image_function(self, image):
        if not with_vigra and self.label_segmentation:
            raise NotImplementedError("Connected components is not supported without vigra "
                                      "if label_segmentation is set to True.")
        if self.label_segmentation:
            connected_components = vigra.analysis.labelImageWithBackground(image)
        else:
            connected_components, _ = label(image)
        return connected_components


class ConnectedComponents3D(Transform):
    """
    Apply connected components on segmentation in 3D.
    """
    def __init__(self, label_segmentation=True, **super_kwargs):
        """
        Parameters
        ----------
        label_segmentation : bool
            Whether the input is a segmentation. If True (default), this computes a
            connected components on both segmentation and binary images (instead of just binary
            images, when this is set to False). However, this would require vigra as a dependency.
        super_kwargs : dict
            Keyword arguments to the super class.
        """
        super(ConnectedComponents3D, self).__init__(**super_kwargs)
        self.label_segmentation = label_segmentation

    def volume_function(self, volume):
        if not with_vigra and self.label_segmentation:
            raise NotImplementedError("Connected components is not supported without vigra "
                                      "if label_segmentation is set to True.")
        if self.label_segmentation:
            connected_components = vigra.analysis.labelVolumeWithBackground(volume.astype('uint32'))
        else:
            connected_components, _ = label(volume)
        return connected_components


# TODO separate Shift Transformation to it's own class
class ManySegmentationsToFuzzyAffinities(Transform):
    """ Crop patch of size `size` from the center of the image """
    def __init__(self, dim, offsets, add_singleton_channel_dimension=True,
                 use_gpu=False, retain_segmentation=False, shift_input=False,
                 multi_scale_factor=None, **super_kwargs):
        super(ManySegmentationsToFuzzyAffinities, self).__init__(**super_kwargs)
        self.dim = dim
        self.shift_input = shift_input
        self.add_singleton_channel_dimension = add_singleton_channel_dimension
        self.use_gpu = use_gpu
        self.retain_segmentation = retain_segmentation
        self.set_new_offset(offsets)
        self.multi_scale_factor = multi_scale_factor


    def single_scale_batch_function(self, image):
        # calculate the affinities for all label image channels
        # and return average
        assert(len(image[1].shape) == self.dim+1)
        affinities = np.sum([self.s2afo(i) for i in image[1]], axis=0)
        affinities /= len(image[1])

        if self.retain_segmentation:
            affinities = np.concatenate((image[1][0:1], affinities), 0)

        if self.shift_input:
            assert(len(self.offsets) == 1)
            return shift_tensor(image[0], self.offsets[0]), affinities
        else:
            return image[0], affinities
        
    def batch_function(self, image):
        if self.multi_scale_factor is None:
            return self.single_scale_batch_function(image)
        else:
            msf = self.multi_scale_factor
            ms = [1, msf**1, msf**2, msf**3]
            return image[0], tuple(self.single_scale_batch_function((image[0][:, ::s, ::s], image[1][:, ::s, ::s]))[1] for s in ms)

    def set_new_offset(self, offsets):
        self.offsets = offsets
        self.s2afo = Segmentation2AffinitiesFromOffsets(self.dim, self.offsets,
            add_singleton_channel_dimension=self.add_singleton_channel_dimension,
            use_gpu=self.use_gpu,
            retain_segmentation=False)


