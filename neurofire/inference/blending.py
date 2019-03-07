import numpy as np


# Blending similar to what is done in
# 'Superhuman accuracy on SNEMI'
# However, the function in there probably contains a
# typo and doesn't make sense
# -> for now we use a linear ramp

# TODO more blending modes ?!
# -> this would be the sub-class `LinearBlending`
# -> if so make a top class and inherit from it for each mode
class Blending(object):

    epsilon = 0.001

    def __init__(self, dim, ramp_size):
        assert dim in (2, 3), str(dim)
        self.dim = dim
        assert isinstance(ramp_size, (int, list, tuple))
        if isinstance(ramp_size, int):
            self.ramp_size = [ramp_size] * self.dim
        else:
            self.ramp_size = list(ramp_size)
        assert len(self.ramp_size) == self.dim

    @staticmethod
    def blending_profile_1d(x, dim_size, ramp_size):
        # In 1-D case, x must resemble an arange
        ramp = np.logical_and(x >= ramp_size, x < (dim_size - ramp_size))
        ramp = np.where(ramp, 1., 0.)
        ramp_up = np.where(x < ramp_size, 1., 0.)
        ramp_down = np.where(x >= (dim_size - ramp_size), 1., 0.)
        profile = (ramp +
                   (1 - ramp) * ramp_up * (x / ramp_size) +
                   (1 - ramp) * ramp_down * ((dim_size - x - 1) / ramp_size))
        return profile

    def _get_blending_mask_2d(self, shape):
        yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        profile_x = self.blending_profile_1d(xx, shape[0], self.ramp_size[0])
        profile_y = self.blending_profile_1d(yy, shape[1], self.ramp_size[1])
        return 0.5 * (profile_x + profile_y)

    def _get_blending_mask_3d(self, shape):
        zz, yy, xx = np.meshgrid(np.arange(shape[0]),
                                 np.arange(shape[1]),
                                 np.arange(shape[2]),
                                 indexing='ij')
        profile_z = self.blending_profile_1d(zz, shape[0], self.ramp_size[0])
        profile_y = self.blending_profile_1d(yy, shape[1], self.ramp_size[1])
        profile_x = self.blending_profile_1d(xx, shape[2], self.ramp_size[2])
        return (profile_x + profile_y + profile_z) / 3.

    def get_blending_mask(self, shape):
        blend_mask = self._get_blending_mask_2d(shape) if self.dim == 2 else \
            self._get_blending_mask_3d(shape)
        # we clip to avoid division by zero later
        return np.clip(blend_mask, self.epsilon, 1.)

    def __call__(self, input_):
        assert input_.ndim in (self.dim, self.dim + 1), '%i, %i' % (input_.ndim, self.dim)
        shape = input_.shape if input_.ndim == self.dim else input_.shape[1:]
        blend_mask = self.get_blending_mask(shape)
        return (blend_mask * input_).astype(input_.dtype), blend_mask
