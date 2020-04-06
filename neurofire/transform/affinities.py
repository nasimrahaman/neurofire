import numpy as np
import inferno.utils.python_utils as pyu
from inferno.io.transform import Transform
from ..criteria.multi_scale_loss import Downsampler
from .segmentation import DtypeMapping

try:
    from affogato.affinities import compute_multiscale_affinities, compute_affinities
except ImportError:
    compute_affinities, compute_multiscale_affinities = None, None


# TODO add more options (membrane prediction)
# helper function that returns affinity transformation from config
def affinity_config_to_transform(**affinity_config):
    assert ('offsets' in affinity_config) != ('block_shapes' in affinity_config), \
        "Need either 'offsets' or 'block_shapes' parameter in config"

    if 'offsets' in affinity_config:
        # whether to calculating affinities on 2D or 3D
        if len(affinity_config['offsets'][0]) == 2:
            return Segmentation2Affinities2D(**affinity_config)
        else:
            return Segmentation2Affinities(**affinity_config)
    else:
        return Segmentation2MultiscaleAffinities(**affinity_config)


class Segmentation2Affinities2or3D(Transform, DtypeMapping):
    def __init__(self, offsets, dtype='float32',
                 retain_mask=False, ignore_label=None,
                 retain_segmentation=False, segmentation_to_binary=False,
                 map_to_foreground=True, learn_ignore_transitions=False,
                 **super_kwargs):
        assert compute_affinities is not None,\
            "Couldn't find 'affogato' module, affinity calculation is not available"
        assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        super(Segmentation2Affinities2or3D, self).__init__(**super_kwargs)
        self.dim = len(offsets[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(off) == self.dim for off in offsets[1:])
        self.offsets = offsets
        self.dtype = dtype
        self.retain_mask = retain_mask
        self.ignore_label = ignore_label
        self.retain_segmentation = retain_segmentation
        self.segmentation_to_binary = segmentation_to_binary
        assert not (self.retain_segmentation and self.segmentation_to_binary),\
            "Currently not supported"
        self.map_to_foreground = map_to_foreground
        self.learn_ignore_transitions = learn_ignore_transitions

    def to_binary_segmentation(self, tensor):
        assert self.ignore_label != 0, "We assume 0 is background, not ignore label"
        if self.map_to_foreground:
            return (tensor == 0).astype(self.dtype)
        else:
            return (tensor != 0).astype(self.dtype)

    def include_ignore_transitions(self, affs, mask, seg):
        ignore_seg = (seg == self.ignore_label).astype(seg.dtype)
        ignore_transitions, invalid_mask = compute_affinities(ignore_seg, self.offsets)
        invalid_mask = np.logical_not(invalid_mask)
        # NOTE affinity convention returned by affogato:
        # transitions are marked by 0
        ignore_transitions = ignore_transitions == 0
        ignore_transitions[invalid_mask] = 0
        affs[ignore_transitions] = 0
        mask[ignore_transitions] = 1
        return affs, mask

    def input_function(self, tensor):
        # print("affs: in shape", tensor.shape)
        if self.ignore_label is not None:
            # output.shape = (C, Z, Y, X)
            output, mask = compute_affinities(tensor, self.offsets,
                                              ignore_label=self.ignore_label,
                                              have_ignore_label=True)
            if self.learn_ignore_transitions:
                output, mask = self.include_ignore_transitions(output, mask, tensor)
        else:
            output, mask = compute_affinities(tensor, self.offsets)

        # FIXME what does this do, need to refactor !
        # hack for platyneris data
        platy_hack = False
        if platy_hack:
            chan_mask = mask[1].astype('bool')
            output[0][chan_mask] = np.min(output[:2], axis=0)[chan_mask]

            chan_mask = mask[2].astype('bool')
            output[0][chan_mask] = np.minimum(output[0], output[2])[chan_mask]

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)
        #
        # print("affs: shape before binary", output.shape)
        if self.segmentation_to_binary:
            output = np.concatenate((self.to_binary_segmentation(tensor)[None],
                                     output), axis=0)
        # print("affs: shape after binary", output.shape)

        # print("affs: shape before mask", output.shape)
        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            mask = mask.astype(self.dtype, copy=False)
            if self.segmentation_to_binary:
                if self.ignore_label is None:
                    additional_mask = np.ones((1,) + tensor.shape, dtype=self.dtype)
                else:
                    additional_mask = (tensor[None] != self.ignore_label).astype(self.dtype)
                mask = np.concatenate([additional_mask, mask], axis=0)
            output = np.concatenate((output, mask), axis=0)
        # print("affs: shape after mask", output.shape)

        # We might want to carry the segmentation along for validation.
        # If this is the case, we insert it before the targets.
        if self.retain_segmentation:
            # Add a channel axis to tensor to make it (C, Z, Y, X) before cating to output
            output = np.concatenate((tensor[None].astype(self.dtype, copy=False), output),
                                    axis=0)

        # print("affs: out shape", output.shape)
        return output


class Segmentation2Affinities(Segmentation2Affinities2or3D):
    def __init__(self, **super_kwargs):
        super(Segmentation2Affinities, self).__init__(**super_kwargs)

    def volume_function(self, tensor):
        assert tensor.ndim == 3
        output = self.input_function(tensor)
        return output


class Segmentation2Affinities2D(Segmentation2Affinities2or3D):
    def __init__(self, **super_kwargs):
        super(Segmentation2Affinities2D, self).__init__(**super_kwargs)

    def image_function(self, tensor):
        assert tensor.ndim == 2
        output = self.input_function(tensor)
        return output


class Segmentation2MultiscaleAffinities(Transform, DtypeMapping):
    def __init__(self, block_shapes, dtype='float32', ignore_label=None,
                 retain_mask=False, retain_segmentation=False,
                 original_scale_offsets=None, **super_kwargs):
        super(Segmentation2MultiscaleAffinities, self).__init__(**super_kwargs)
        assert compute_multiscale_affinities is not None,\
            "Couldn't find 'affogato' module, affinity calculation is not available"
        assert pyu.is_listlike(block_shapes)
        self.block_shapes = block_shapes
        self.dim = len(block_shapes[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(bs) == self.dim for bs in block_shapes[1:])

        self.dtype = dtype
        self.ignore_label = ignore_label
        self.retain_mask = retain_mask
        self.retain_segmentation = retain_segmentation
        self.original_scale_offsets = original_scale_offsets
        if self.retain_segmentation:
            self.downsamplers = [Downsampler(bs) for bs in self.block_shapes]

    def tensor_function(self, tensor):
        # for 2 d input, we need singleton input
        if self.dim == 2:
            assert tensor.shape[0] == 1
            tensor = tensor[0]

        outputs = []
        for ii, bs in enumerate(self.block_shapes):
            # if the block shape is all ones, we can compute normal affinities
            # with nearest neighbor offsets. This should yield the same result,
            # but should be more efficient.
            original_scale = all(s == 1 for s in bs)
            if original_scale:
                if self.original_scale_offsets is None:
                    offsets = [[0 if i != d else -1 for i in range(self.dim)]
                               for d in range(self.dim)]
                else:
                    offsets = self.original_scale_offsets
                output, mask = compute_affinities(tensor.squeeze().astype('uint64'), offsets,
                                                  ignore_label=0 if self.ignore_label is None else self.ignore_label,
                                                  have_ignore_label=False if self.ignore_label is None else True)
            else:
                output, mask = compute_multiscale_affinities(tensor.squeeze().astype('uint64'), bs,
                                                             ignore_label=0 if self.ignore_label is None
                                                             else self.ignore_label,
                                                             have_ignore_label=False if self.ignore_label is None
                                                             else True)

            # Cast to be sure
            if not output.dtype == self.dtype:
                output = output.astype(self.dtype)

            # We might want to carry the mask along.
            # If this is the case, we insert it after the targets.
            if self.retain_mask:
                output = np.concatenate((output, mask.astype(self.dtype, copy=False)), axis=0)
            # We might want to carry the segmentation along for validation.
            # If this is the case, we insert it before the targets for the original scale.
            if self.retain_segmentation:
                ds_target = self.downsamplers[ii](tensor.astype(self.dtype, copy=False))
                if ds_target.ndim != output.ndim:
                    assert ds_target.ndim == output.ndim - 1
                    ds_target = ds_target[None]
                output = np.concatenate((ds_target, output), axis=0)
            outputs.append(output)

        return outputs
