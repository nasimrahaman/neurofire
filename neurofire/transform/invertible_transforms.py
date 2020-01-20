# Invertible transformations for test time da
import math
from copy import deepcopy
import numpy as np


class InvertibleTransform(object):

    # similar to logic in `Transform`, but so far we only have `image_function` and `volume_function`
    # that should be the only thing we need for TestTime DA
    def apply(self, tensor):
        assert tensor.ndim in (2, 3, 4)

        # apply the volume function if implemented
        if hasattr(self, 'volume_function'):
            if tensor.ndim == 4:
                transformed = np.array([self.volume_function(vol) for vol in tensor])
            elif tensor.ndim == 3:
                transformed = self.volume_function(tensor)
            else:
                raise AttributeError("Volume function cannot be applied to 2d array")

        # apply the image function if implemented
        elif hasattr(self, 'image_function'):
            if tensor.ndim == 4:
                transformed = np.array([np.array([self.image_function(image)
                                        for image in channel_image]) for channel_image in tensor])
            elif tensor.ndim == 3:
                transformed = np.array([self.image_function(image) for image in tensor])
            else:
                transformed = self.image_function(tensor)
        else:
            raise AttributeError("Invalid invertible transormation")

        # return the transformed
        return transformed

    def invert(self, tensor):
        assert tensor.ndim in (2, 3, 4)

        # apply the inverse volume function if implemented
        if hasattr(self, 'inverse_volume_function'):
            if tensor.ndim == 4:
                transformed = np.array([self.inverse_volume_function(vol) for vol in tensor])
            elif tensor.ndim == 3:
                transformed = self.inverse_volume_function(tensor)
            else:
                raise AttributeError("Inverse volume function cannot be applied to 2d array")

        # apply the image function if implemented
        elif hasattr(self, 'inverse_image_function'):
            if tensor.ndim == 4:
                transformed = np.array([np.array([self.inverse_image_function(image)
                                        for image in channel_image]) for channel_image in tensor])
            elif tensor.ndim == 3:
                transformed = np.array([self.inverse_image_function(image) for image in tensor])
            else:
                transformed = self.inverse_image_function(tensor)
        else:
            raise AttributeError("Invalid invertible transormation")

        # return the transformed
        return transformed

    # apply to channel offsets
    def apply_offset(self, offsets):
        assert isinstance(offsets, tuple), str(offsets)
        assert len(offsets) in (2, 3)
        if hasattr(self, 'apply_offset_2d'):
            if len(offsets) == 2:
                return self.apply_offset_2d(offsets)
            else:
                # For a 3d tensor we apply the 2d transformation only along yx (last to axes)
                return (offsets[0],) + self.apply_offset_2d(offsets[1:])
        elif hasattr(self, 'apply_offset_3d'):
            assert len(offsets) == 3
            return self.apply_offset_3d(offsets)
        else:
            raise AttributeError("Invalid invertible transormation")

    # apply inverse trafo to channel offsets
    def invert_offset(self, offsets):
        assert isinstance(offsets, tuple), str(offsets)
        assert len(offsets) in (2, 3)
        if hasattr(self, 'invert_offset_2d'):
            if len(offsets) == 2:
                return self.invert_offset_2d(offsets)
            else:
                # For a 3d tensor we apply the 2d transformation only along yx (last to axes)
                return (offsets[0],) + self.invert_offset_2d(offsets[1:])
        elif hasattr(self, 'invert_offset_3d'):
            assert len(offsets) == 3
            return self.invert_offset_3d(offsets)
        else:
            raise AttributeError("Invalid invertible transormation")


class ComposeInvertibles(object):
    """Composes multiple invertible transformations."""
    def __init__(self, *transforms):
        """
        Parameters
        ----------
        transforms : list of invertible transforms to compose.
        """
        assert all([isinstance(trafo, (InvertibleTransform, ComposeInvertibles)) for trafo in transforms]),\
            str(transforms)
        self.transforms = list(transforms)

    def add(self, transform):
        assert isinstance(transform, InvertibleTransform)
        self.transforms.append(transform)
        return self

    def remove(self, name):
        transform_idx = None
        for idx, transform in enumerate(self.transforms):
            if type(transform).__name__ == name:
                transform_idx = idx
                break
        if transform_idx is not None:
            self.transforms.pop(transform_idx)
        return self

    def apply(self, tensor):
        intermediate = tensor
        for transform in self.transforms:
            intermediate = transform.apply(intermediate)
        return intermediate

    def invert(self, tensor):
        intermediate = tensor
        # when we invert transforms, we need to apply the inverse
        # in reverse order
        for transform in self.transforms[::-1]:
            intermediate = transform.invert(intermediate)
        return intermediate

    def apply_offset(self, offsets):
        new_offsets = deepcopy(offsets)
        for transform in self.transforms:
            new_offsets = transform.apply_offset(new_offsets)
        return new_offsets

    def invert_offset(self, offsets):
        new_offsets = deepcopy(offsets)
        for transform in self.transforms[::-1]:
            new_offsets = transform.invert_offset(new_offsets)
        return new_offsets


class InvertibleRotation(InvertibleTransform):
    def __init__(self, n_rotations):
        assert 0 < n_rotations < 4
        self.n_rotations = n_rotations
        self.inverse_rotations = 4 - self.n_rotations

    def image_function(self, image):
        return np.rot90(image, k=self.n_rotations)

    def inverse_image_function(self, image):
        return np.rot90(image, k=self.inverse_rotations)

    def rotate_offset(self, offset, k):
        # make offset float
        x, y = [float(off) for off in offset]
        # the proper angle for the number of rotations FIXME times minus or not?
        a = k * math.pi / 2.0
        # the rotated offset
        rot_x = x * math.cos(a) - y * math.sin(a)
        rot_y = x * math.sin(a) + y * math.cos(a)
        rot_x = round(rot_x, 4)
        rot_y = round(rot_y, 4)
        assert rot_x.is_integer()
        assert rot_y.is_integer()
        return int(rot_x), int(rot_y)

    def apply_offset_2d(self, offset):
        return self.rotate_offset(offset, self.n_rotations)

    def invert_offset_2d(self, offset):
        return self.rotate_offset(offset, self.inverse_rotations)


class InvertibleFlip2D(InvertibleTransform):
    def __init__(self, flip_dim):
        assert flip_dim < 2
        self.flip_dim = flip_dim

    def image_function(self, image):
        if self.flip_dim == 0:
            image = image[::-1]
        else:
            image = image[:, ::-1]
        return image

    def inverse_image_function(self, image):
        return self.image_function(image)

    def apply_offset_2d(self, offsets):
        assert len(offsets) == 2
        # y - flip -> y offset is inverted
        if self.flip_dim == 0:
            return (-offsets[0], offsets[1])
        # x - flip -> x offset is inverted
        elif self.flip_dim == 1:
            return (offsets[0], -offsets[1])

    def invert_offset_2d(self, offsets):
        return self.apply_offset_2d(offsets)


class InvertibleFlip3D(InvertibleTransform):
    def __init__(self, flip_dim):
        assert flip_dim < 3
        self.flip_dim = flip_dim

    def volume_function(self, volume):
        if self.flip_dim == 0:
            volume = volume[::-1]
        elif self.flip_dim == 1:
            volume = volume[:, ::-1]
        else:
            volume = volume[:, :, ::-1]
        return volume

    def inverse_volume_function(self, volume):
        return self.volume_function(volume)

    def apply_offset_3d(self, offsets):
        assert len(offsets) == 3
        # z - flip -> z offset is inverted
        if self.flip_dim == 0:
            return (-offsets[0], offsets[1], offsets[2])
        # y - flip -> y offset is inverted
        elif self.flip_dim == 1:
            return (offsets[0], -offsets[1], offsets[2])
        elif self.flip_dim == 2:
            return (offsets[0], offsets[1], -offsets[2])

    def invert_offset_3d(self, offsets):
        return self.apply_offset_3d(offsets)


class InvertibleTranspose2D(InvertibleTransform):
    def __init__(self, do_transpose=True):
        self.do_transpose = do_transpose

    def image_function(self, image):
        return image.transpose() if self.do_transpose else image

    def inverse_image_function(self, image):
        return self.image_function(image)

    def apply_offset_2d(self, offsets):
        assert len(offsets) == 2
        return offsets[::-1] if self.do_transpose else offsets

    def invert_offset_2d(self, offsets):
        return self.apply_offset_2d(offsets)


class InvertibleTranspose3D(InvertibleTransform):
    def __init__(self, do_transpose=True):
        self.do_transpose = do_transpose

    def volume_function(self, volume):
        return volume.transpose() if self.do_transpose else volume

    def inverse_volume_function(self, volume):
        return self.volume_function(volume)

    def apply_offset_3d(self, offsets):
        assert len(offsets) == 3
        return offsets[::-1] if self.do_transpose else offsets

    def invert_offset_3d(self, offsets):
        return self.apply_offset_3d(offsets)
