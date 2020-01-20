import numpy as np
from ..transform.invertible_transforms import InvertibleTransform, ComposeInvertibles
from ..transform.invertible_transforms import InvertibleFlip2D, InvertibleFlip3D, InvertibleRotation
from ..transform.invertible_transforms import InvertibleTranspose2D, InvertibleTranspose3D


# this returns a 2d array with the all the indices of matching rows for a and b
# cf. http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
def find_matching_row_indices(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # using a dictionary, this is faster than the pure np variant
    indices = []
    rows_x = {tuple(row): i for i, row in enumerate(x)}
    for i, row in enumerate(y):
        if tuple(row) in rows_x:
            indices.append([rows_x[tuple(row)], i])
    return np.array(indices)


class TestTimeAugmenter(object):
    combination_modes = ('mean', 'max', 'min')

    def __init__(self, transforms, combination_mode='mean'):
        assert isinstance(transforms, (list, tuple))
        np.all(isinstance(trafo, (InvertibleTransform, ComposeInvertibles)) for trafo in transforms)
        self.transforms = list(transforms)
        assert combination_mode in self.combination_modes or isinstance(combination_mode, dict), type(combination_mode)
        if isinstance(combination_mode, str):
            self.combinator = self.mode_str_to_method(combination_mode)
        # we accept a dict to apply different modes for different channels
        elif isinstance(combination_mode, dict):
            assert combination_mode['combinator1'] in self.combination_modes
            assert combination_mode['combinator2'] in self.combination_modes
            assert isinstance(combination_mode['switch_channel'], int)
            self.combinator = {key: self.mode_str_to_method(val) if val in self.combination_modes else val
                               for key, val in combination_mode.items()}

    @staticmethod
    def mode_str_to_method(combination_mode):
        if combination_mode == 'mean':
            combinator = np.mean
        elif combination_mode == 'max':
            combinator = np.amax
        elif combination_mode == 'min':
            combinator = np.amin  # NOTE min makes sense for actual affinities
        return combinator

    # test-time data augmentation without affinity-offsets
    def _apply_without_offsets(self, input_, inference):
        # apply the model once without augmentations
        # I don't want to apply the identity, because then we would
        # need to infer the out shape somehow
        output = [inference(input_)]
        for trafo in self.transforms:
            # apply the transformation
            tda_input_ = trafo.apply(input_)
            output_tmp = inference(tda_input_)
            # reverse the transformations for the output
            output.append(trafo.invert(output_tmp))
        output = np.array(output)
        return self.combinator(output, axis=0)

    # produce full ring of affinities
    def make_full_affinities(self, affinities, offsets):
        assert affinities.ndim in (3, 4)
        shape = affinities.shape[1:]
        dim = len(shape)
        padding_size = int(np.abs(offsets).max())

        full_affinities = []
        full_offsets = []

        # iterate over the offsets / affinities
        for i_offset, offset in enumerate(offsets):

            # normalize the offset
            offset = tuple(int(off) for off in offset)
            affinity_channel = affinities[i_offset].squeeze()

            # if this is a zero offset, just add it to the full affinities and continue
            if all(off == 0 for off in offset):
                full_affinities.append(affinity_channel[None])
                full_offsets.append(offset)
                continue

            # pad the current affinity channel
            padded_affinity_channel = np.pad(affinity_channel, padding_size, mode='reflect')

            # the point-mirrored  offset
            p_offset = tuple(-1 * val for val in offset)

            # the corresponding affinity image offset
            shift_offsets = [padding_size + p_offset[i] for i in range(dim)]
            shifts = tuple(slice(soff, soff + shape[i]) for i, soff in enumerate(shift_offsets))

            # affinity channel of the point mirrored affinity
            p_affinity = padded_affinity_channel[shifts]

            # append the original affinity and point-mirrored
            full_affinities.append(affinity_channel[None])
            full_offsets.append(offset)
            full_affinities.append(p_affinity[None])
            full_offsets.append(p_offset)

        full_affinities = np.concatenate(full_affinities, axis=0)
        return full_affinities, full_offsets

    # invert transformation on affinity maps
    def invert_affinity_transform(self, affinities, offsets, trafo):
        # dict to find offsets
        offset_dict = {tuple(offset): i_offset for i_offset, offset in enumerate(offsets)}
        # the result
        inverted_affinites = [None] * len(offsets)

        # make full affinities to ensure that we can later map all inverted affinity
        # channels back to an original channel
        full_affinities, full_offsets = self.make_full_affinities(affinities, offsets)

        # iterate over the full affinities
        for off, affinity_channel in zip(full_offsets, full_affinities):

            # caclulate the inverted offset
            inverted_offset = tuple(trafo.invert_offset(off))
            # if the inverted offset is in our original offsets,
            # invert the affinity channel and write it to the correct channel
            if inverted_offset in offset_dict:
                # invert affinity channel
                inverted_affinity_channel = trafo.invert(affinity_channel)
                # find the correct channel index
                offset_index = offset_dict[inverted_offset]
                inverted_affinites[offset_index] = inverted_affinity_channel[None]

        assert all(iva is not None for iva in inverted_affinites), \
            'One or more offset could not be inverted'
        inverted_affinites = np.concatenate(inverted_affinites, axis=0)
        assert inverted_affinites.shape == affinities.shape, "%s, %s" \
            % (str(inverted_affinites.shape), str(affinities.shape))
        return inverted_affinites

    # test-time data augmentation with affinity-offsets
    def _apply_with_offsets(self, input_, inference, offsets):
        # apply the model once without augmentations
        # I don't want to apply the identity, because then we would
        # need to infer the out shape somehow
        output = [inference(input_)]
        for trafo in self.transforms:
            # apply the transformation to input and predict
            tda_input_ = trafo.apply(input_)
            output_tmp = inference(tda_input_)
            # invert the affinity and append to output
            output.append(self.invert_affinity_transform(output_tmp, offsets, trafo))
        output = np.array(output)
        if isinstance(self.combinator, dict):
            switch_chan = self.combinator['switch_channel']
            combined = np.concatenate(
                [self.combinator['combinator1'](output[:, :switch_chan], axis=0),
                 self.combinator['combinator2'](output[:, switch_chan:], axis=0)], axis=0
            )
            return combined
        else:
            return self.combinator(output, axis=0)

    def __call__(self, input_, inference, offsets=None):
        if offsets is None:
            assert callable(self.combinator), \
                "Multiple combination modes are not supported for single channel inference"
            return self._apply_without_offsets(input_, inference)
        else:
            assert isinstance(offsets, (list, tuple))
            if isinstance(self.combinator, dict):
                assert self.combinator['switch_channel'] < len(offsets)
            return self._apply_with_offsets(input_, inference, offsets)

    @classmethod
    def default_tda(cls, dim, combination_mode='mean'):
        transforms = cls.default_transformations(dim)
        return cls(transforms, combination_mode)

    @staticmethod
    def default_transformations(dim):
        assert dim in (2, 3)
        return TestTimeAugmenter._default_transformations_2d() if dim == 2 else \
            TestTimeAugmenter._default_transformations_3d()

    # return all independent 2d transformations
    @staticmethod
    def _default_transformations_2d():
        default_transformations = [ComposeInvertibles(InvertibleFlip2D(flip_dim),
                                                      InvertibleRotation(n_rot),
                                                      InvertibleTranspose2D(do_transpose))
                                   for flip_dim in (0, 1) for n_rot in (1, 2, 3) for do_transpose in (True, False)
                                   if not ((flip_dim == 0 and n_rot == 1 and do_transpose) or
                                           (flip_dim == 0 and n_rot == 1 and not do_transpose) or
                                           (flip_dim == 1 and n_rot == 1 and do_transpose) or
                                           (flip_dim == 1 and n_rot == 1 and not do_transpose) or
                                           (flip_dim == 0 and n_rot == 3 and do_transpose)
                                           )]
        return default_transformations

    @staticmethod
    def default_trafo_names_2d():
        default_names = ["Flip2D_%i_Rotations_%i_Transpose_%s" % (flip_dim, n_rot, str(do_transpose))
                         for flip_dim in (0, 1) for n_rot in (1, 2, 3) for do_transpose in (True, False)
                         if not ((flip_dim == 0 and n_rot == 1 and do_transpose) or
                                 (flip_dim == 0 and n_rot == 1 and not do_transpose) or
                                 (flip_dim == 1 and n_rot == 1 and do_transpose) or
                                 (flip_dim == 1 and n_rot == 1 and not do_transpose) or
                                 (flip_dim == 0 and n_rot == 3 and do_transpose)
                                 )]
        return default_names

    # return all independent 3d transformations
    @staticmethod
    def _default_transformations_3d():
        default_transformations = [ComposeInvertibles(InvertibleFlip3D(flip_dim),
                                                      InvertibleRotation(n_rot),
                                                      InvertibleTranspose3D(do_transpose))
                                   for flip_dim in (0, 1, 2) for n_rot in (1, 2, 3) for do_transpose in (True, False)
                                   if not((flip_dim == 1 and n_rot == 1 and do_transpose) or
                                          (flip_dim == 1 and n_rot == 1 and not do_transpose) or
                                          (flip_dim == 2 and n_rot == 1 and do_transpose) or
                                          (flip_dim == 2 and n_rot == 1 and not do_transpose))]
        return default_transformations

    @staticmethod
    def default_trafo_names_3d():
        default_names = ["Flip3D_%i_Rotations_%i_Transpose_%s" % (flip_dim, n_rot, str(do_transpose))
                         for flip_dim in (0, 1, 2) for n_rot in (1, 2, 3) for do_transpose in (True, False)
                         if not((flip_dim == 1 and n_rot == 1 and do_transpose) or
                                (flip_dim == 1 and n_rot == 1 and not do_transpose) or
                                (flip_dim == 2 and n_rot == 1 and do_transpose) or
                                (flip_dim == 1 and n_rot == 3 and do_transpose) or
                                (flip_dim == 2 and n_rot == 1 and not do_transpose))]
        return default_names
