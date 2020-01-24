import numpy as np

from inferno.io.transform import Transform
from scipy.ndimage.morphology import binary_dilation


class RandomSlide(Transform):
    """Transform to randomly sample misalignments."""
    def __init__(self, output_image_size=None, max_misalign=None,
                 targets_have_defected_mask=False,
                 defected_label=None,
                 shift_vs_slide_proba=0.5, apply_proba=0.8, **super_kwargs):
        """
        Shift: move only one slice
        Slide: from that point on, move all slices
        shift_vs_slide_proba: If less then this, shift. If more, slide
        """
        # Make sure that output image size and max misalign are not both None
        assert not (output_image_size is None and max_misalign is None)
        # Make sure that output image size and max misalign are not set both
        assert not (output_image_size is not None and max_misalign is not None)

        if output_image_size is not None:
            output_image_size = (output_image_size, output_image_size) \
                if isinstance(output_image_size, int) else output_image_size
            assert len(output_image_size) == 2, \
                "Inconsistent output_image_size: {}. Must have 2 elements.".format(output_image_size)
            self.output_image_size = tuple(output_image_size)
            self.max_misalign = None
        else:
            self.output_image_size = None
            self.max_misalign = (max_misalign, max_misalign) if isinstance(max_misalign, int) \
                else max_misalign
            assert len(self.max_misalign) == 2

        super(RandomSlide, self).__init__(**super_kwargs)
        # Make sure we have a 2-tuple
        self.shift_vs_slide_proba = shift_vs_slide_proba
        self.apply_proba = apply_proba
        self.targets_have_defected_mask = targets_have_defected_mask
        self.defected_label = defected_label

    def build_random_variables(self, num_planes, input_image_size):
        # Compute the available slide leeways. We have two sets of leeways - origin-ward
        # (i.e. close to the origin at top left of an image) and antiorigin-ward
        # (closer to the bottom right)
        out_size = self.output_image_size if self.output_image_size is not None else \
            tuple(imshape - maxmis for imshape, maxmis in zip(input_image_size, self.max_misalign))

        originward_leeways = tuple((_input_size - _output_size) // 2
                                   for _input_size, _output_size in zip(input_image_size, out_size))
        antioriginward_leeways = tuple(_input_size - _output_size - _originward
                                       for _input_size, _output_size, _originward in
                                       zip(input_image_size, out_size, originward_leeways))
        # We have: leeways[0] = (originward, antioriginward)
        leeways = tuple(zip(originward_leeways, antioriginward_leeways))
        # Now to sample the shifts, we fix our origin on the top left corner of the input image.
        # From there on, we could go a max orginward_leeway to the left or antioriginward_leeway to
        # the right.
        shifts = tuple(np.random.randint(low=-leeway[0], high=leeway[1]) for leeway in leeways)
        # Select whether to shift or slide
        shift_or_slide = 'shift' if np.random.uniform() < self.shift_vs_slide_proba else 'slide'
        # Select from which plane on to slide
        slide_from = np.random.randint(low=1, high=num_planes)
        shift_at = np.random.randint(low=1, high=num_planes)
        # Write to dict
        self.set_random_variable('shifts', shifts)
        self.set_random_variable('origin', originward_leeways)
        self.set_random_variable('shift_or_slide', shift_or_slide)
        self.set_random_variable('slide_from', slide_from)
        self.set_random_variable('shift_at', shift_at)

    def shift_and_crop(self, image, zero_shift=False):
        # Get random variables
        origin = self.get_random_variable('origin')
        shifts = self.get_random_variable('shifts')
        # Kill shift if requested
        if zero_shift:
            shifts = (0, 0)
        out_size = self.output_image_size if self.output_image_size is not None else \
            tuple(imshape - maxmis for imshape, maxmis in zip(image.shape, self.max_misalign))
        # Get slice
        starts = tuple(_origin + _shift for _origin, _shift in zip(origin, shifts))
        stops = tuple(_start + _size for _start, _size in zip(starts, out_size))
        slices = tuple(slice(_start, _stop) for _start, _stop in zip(starts, stops))
        # Crop and return
        return image[slices]

    def batch_function(self, tensors):
        assert isinstance(tensors, (tuple, list))
        tensors = list(tensors)

        # Add channel dim to all tensors (if not present already):
        assert all([vol.ndim in [3,4] for vol in tensors])
        tensors = [vol if vol.ndim == 4 else np.expand_dims(vol, axis=0) for vol in tensors]

        # FIXME: generalize to multiple tensors...
        # I need to handle defected mask, shift should not change GT, etc...
        assert len(tensors) == 2
        defected_mask = None
        if self.targets_have_defected_mask:
            assert self.defected_label is not None
            defected_mask = tensors[1][1] == self.defected_label
            defected_mask = defected_mask.max(axis=-1).max(axis=-1)
            # We should not slide/shift any neighboring slice to one defected:
            defected_mask = binary_dilation(defected_mask)

        vol_shape = tensors[0].shape[1:]
        if len(tensors) > 1:
            assert all(tnsr.shape[1:] == vol_shape for tnsr in tensors[1:]), "%s" % ", ".join(str(tnsr.shape) for tnsr in tensors)
        # Build random variables
        self.build_random_variables(num_planes=vol_shape[0],
                                    input_image_size=vol_shape[1:])

        # determine if we apply the transformation to the slide at all
        # TODO would be cleaner to integrate into `build random variables` as well
        apply_shift = np.random.rand() < self.apply_proba

        apply_to = range(len(tensors)) if self._apply_to is None else self._apply_to

        if apply_shift:
            # Check if the slice is not

            # Get random variables
            shift_or_slide = self.get_random_variable('shift_or_slide')
            # Shift or slide?
            if shift_or_slide == 'shift':
                # Shift
                shift_at = self.get_random_variable('shift_at')
                is_defected = defected_mask[shift_at] if defected_mask is not None else False

                if not is_defected:
                    # Don't shift if plane_num doesn't equal shift_at
                    return tuple(np.array([[self.shift_and_crop(image=plane,
                                                                      zero_shift=(plane_num != shift_at) or
                                                                                 (nb_tnsr not in apply_to))
                                                 for plane_num, plane in enumerate(vv)] for vv in tnsr])
                                 for nb_tnsr, tnsr in enumerate(tensors))
            else:
                # Slide
                slide_from = self.get_random_variable('slide_from')
                is_defected = defected_mask[slide_from] if defected_mask is not None else False

                if not is_defected:
                    # Don't shift if plane_num isn't larger than or equal to slide_from
                    return tuple(np.array([[self.shift_and_crop(image=plane,
                                                                      zero_shift=(plane_num < slide_from))
                                                 for plane_num, plane in enumerate(vv)] for vv in tnsr]) for nb_tnsr, tnsr in enumerate(tensors))

        return tuple(np.array([[self.shift_and_crop(image=plane, zero_shift=True)
                                         for plane in vv] for vv in tnsr]) for tnsr in tensors)


class RejectNonZeroThreshold(object):
    def __init__(self, threshold):
        self.threshold = threshold

    # return True if ration of non-zeros is below the threshold and hence the batch should be rejected
    def __call__(self, fetched):
        return (np.count_nonzero(fetched) / fetched.size) < self.threshold


class Normalize01(Transform):
    def __init__(self, eps=1.e-4, **super_kwargs):
        super(Normalize01, self).__init__(**super_kwargs)
        self.eps = eps

    def tensor_function(self, tensor):
        min_ = tensor.min()
        max_ = tensor.max()
        tensor = (tensor - min_) / (max_ + self.eps)
        return tensor
