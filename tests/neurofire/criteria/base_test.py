import unittest
import numpy as np


class BaseTest(unittest.TestCase):
    shape = (32, 32, 32)

    @staticmethod
    def make_segmentation_with_ignore(shape):
        seg = np.zeros(shape, dtype='int32')
        label = 1
        # make randoms segmentation
        for index in range(seg.size):
            coord = np.unravel_index(index, shape)
            seg[coord] = label
            if np.random.rand() > 0.9:
                label += 1
        # mask 10 % random pixel
        mask = np.random.choice([0, 1],
                                size=seg.size,
                                p=[0.9, 0.1]).reshape(seg.shape).astype('bool')
        seg[mask] = 0
        return seg

    @staticmethod
    def make_segmentation(shape):
        seg = np.zeros(shape, dtype='int32')
        label = 1
        # make randoms segmentation
        for index in range(seg.size):
            coord = np.unravel_index(index, shape)
            seg[coord] = label
            if np.random.rand() > 0.9:
                label += 1
        return seg

    @staticmethod
    def brute_force_transition_masking(segmentation, offsets, ignore_value=0):
        # squeeze away the batch dimension
        segmentation = segmentation.squeeze()
        ndim = segmentation.ndim
        shape = segmentation.shape
        n_channels = len(offsets)
        mask_shape = (n_channels,) + shape
        mask = np.zeros(mask_shape, dtype='bool')

        # get the strides (need to divide by bytesize)
        byte_size = np.dtype(mask.dtype).itemsize
        strides = [s // byte_size for s in mask.strides]

        # iterate over all edges (== 1d affinity coordinates) to get the masking values
        for edge in range(mask.size):

            # translate edge-id to affinity coordinate
            coord = (edge // strides[0],)
            coord = coord + tuple((edge % strides[d - 1]) // strides[d] for d in range(1, ndim + 1))

            # get the current offset
            offset = offsets[coord[0]]
            # get the spatial coordinates
            coord_u, coord_v = coord[1:], coord[1:]
            # apply the offset to coord v
            coord_v = tuple(cv + off for cv, off in zip(coord_v, offset))

            # check if coord v is valid, if not, set mask and continue
            if(not all(0 <= cv < sha for cv, sha in zip(coord_v, shape))):
                mask[coord] = True
                continue

            u, v = segmentation[coord_u], segmentation[coord_v]
            if u == ignore_value or v == ignore_value:
                mask[coord] = True

        return mask
