from math import ceil, floor
import inferno.io.volumetric as io
from inferno.io.transform import Compose
from inferno.io.transform.volume import AdditiveNoise
from inferno.io.transform.generic import Cast, Normalize
from inferno.io.core.base import SyncableDataset
from inferno.io.core.base import IndexSpec
from inferno.utils.io_utils import yaml2dict

from neurofire.transform.defect_augmentation import DefectAugmentation
from neurofire.transform.volume import Normalize01

try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False

try:
    from mu_net.transform.superpixel_augmentation import WatershedAugmentation
except ImportError:
    WatershedAugmentation = None


def read_param(param, name, type_):
    if param is None:
        return None
    if isinstance(param, dict):
        assert name is not None
        assert name in param
        return param[name]
    elif isinstance(param, type_):
        return param
    else:
        raise AttributeError("No such param %s" % name)


class RawVolume(io.HDF5VolumeLoader):
    def __init__(self, path, path_in_file=None,
                 data_slice=None, name=None, dtype='float32',
                 mean=None, std=None, sigma=None, zero_mean_unit_variance=True,
                 p_augment_ws=0., **slicing_config):
        super().__init__(path=path, path_in_h5_dataset=path_in_file,
                         data_slice=data_slice, name=name, **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        if name is not None:
            mean = mean[name] if isinstance(mean, dict) else mean
            std = std[name] if isinstance(std, dict) else std
            sigma = sigma[name] if isinstance(sigma, dict) else sigma
            zero_mean_unit_variance = zero_mean_unit_variance[name] if isinstance(zero_mean_unit_variance, dict)\
                else zero_mean_unit_variance
            p_augment_ws = p_augment_ws[name] if isinstance(p_augment_ws, dict) else p_augment_ws

        self.transforms = self.get_transforms(mean, std, sigma,
                                              p_augment_ws, zero_mean_unit_variance)

    def get_transforms(self, mean, std, sigma,
                       p_augment_ws, zero_mean_unit_variance):
        transforms = Compose(Cast(self.dtype))
        # add normalization (zero mean / unit variance)
        if zero_mean_unit_variance:
            transforms.add(Normalize(mean=mean, std=std))
        else:
            transforms.add(Normalize01())
        # add noise transform if specified
        if sigma is not None:
            transforms.add(AdditiveNoise(sigma=sigma))
        # add watershed super-pixel augmentation is specified
        if p_augment_ws > 0.:
            assert WatershedAugmentation is not None
            transforms.add(WatershedAugmentation(p_augment_ws, invert=True))
        return transforms


class N5RawVolume(io.LazyN5VolumeLoader):
    def __init__(self, path, path_in_file=None,
                 data_slice=None, name=None, dtype='float32',
                 mean=None, std=None, sigma=None,
                 zero_mean_unit_variance=True, p_augment_ws=0,
                 **slicing_config):
        super().__init__(path=path, path_in_file=path_in_file,
                         data_slice=data_slice, name=name, **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms(mean, std, sigma,
                                              p_augment_ws, zero_mean_unit_variance)

    def get_transforms(self, mean, std, sigma,
                       p_augment_ws, zero_mean_unit_variance):
        transforms = Compose(Cast(self.dtype))
        # add normalization (zero mean / unit variance)
        if zero_mean_unit_variance:
            transforms.add(Normalize(mean=mean, std=std))
        else:
            transforms.add(Normalize01())
        # add noist transform if specified
        if sigma is not None:
            transforms.add(AdditiveNoise(sigma=sigma))
        # add watershed super-pixel augmentation is specified
        if p_augment_ws > 0.:
            assert WatershedAugmentation is not None
            transforms.add(WatershedAugmentation(p_augment_ws, invert=True))
        return transforms


class RawVolumeWithDefectAugmentation(RawVolume):
    def __init__(self, path, defect_augmentation_config,
                 name=None, path_in_file=None,
                 data_slice=None, dtype='float32',
                 ignore_slice_list=None, mean=None, std=None,
                 sigma=None, zero_mean_unit_variance=True,
                 p_augment_ws=0., **slicing_config):

        super().__init__(path=path, path_in_file=path_in_file,
                         data_slice=data_slice, name=name,
                         dtype=dtype, mean=mean, std=std, sigma=sigma,
                         p_augment_ws=p_augment_ws,
                         zero_mean_unit_variance=zero_mean_unit_variance,
                         **slicing_config)

        defect_augmentation_config = yaml2dict(defect_augmentation_config)
        defect_augmentation_config.update({'ignore_slice_list': ignore_slice_list})
        self.defect_augmentation = DefectAugmentation.from_config(defect_augmentation_config)
        self.cast = Cast(self.dtype)

        # Check if we should keep track of defected slices:
        self.keep_track_defected_slices = len(self.defect_augmentation.keep_track_of) > 0

    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = tuple(self.base_sequence[index])
        vol = self.volume[slices]

        # first we need to cast to the proper dtype
        vol = self.cast(vol)

        # next apply the defect transformations
        # NOTE the defect transformations need to be applied before the other transformations,
        # otherwise they screw with the normalization !
        z_offset = slices[0].start
        vol = self.defect_augmentation(vol, z_offset=z_offset)

        # apply the normal transformations (including normalization)
        if self.transforms is not None:
            if self.keep_track_defected_slices:
                if isinstance(vol, tuple):
                    vol = list(vol)
                assert isinstance(vol, list)
                vol[0] = self.transforms(vol[0])
            else:
                vol = self.transforms(vol)

        if self.return_index_spec:
            return vol, IndexSpec(index=index, base_sequence_at_index=slices)
        else:
            return vol

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        path = config.get('path')
        defect_augmentation_config = config.get('defect_augmentation_config')
        path_in_h5_dataset = config.get('path_in_h5_dataset', None)
        data_slice = config.get('data_slice', None)
        name = config.get('name', None)
        dtype = config.get('dtype', 'float32')
        ignore_slice_list = config.get('ignore_slice_list', None)
        slicing_config = config.get('slicing_config', None)
        return cls(path, defect_augmentation_config,
                   path_in_h5_dataset=path_in_h5_dataset,
                   data_slice=data_slice, name=name, dtype=dtype,
                   ignore_slice_list=ignore_slice_list, **slicing_config)


class MultiscaleRawVolume(SyncableDataset):
    def __init__(self, path, paths_in_file,
                 name=None, dtype='float32', mean=None, std=None,
                 scale_factors=None, downscale_policy=None, **slicing_config):
        assert WITH_Z5PY
        assert isinstance(paths_in_file, (list, tuple)), type(paths_in_file)
        self.n_scales = len(paths_in_file)
        assert self.n_scales >= 2

        # the downscale policy determines the shape of the
        # volumes returned at lower scales.
        # e.g. if downscale policy == 1, the shape will
        # be the same as downsampling at the original scale
        # if 0.5 it will be half the shape etc.
        # if None, the downscaled volume will not be changed.
        # downscale_policy can also be a tuple, then the
        # second factor will be the required divisor for the shape
        self.downscale_policy = downscale_policy

        # get the downscaling factors
        if scale_factors is None:
            self.scale_factors = [2**i for i in range(self.n_scales)]
        else:
            assert len(scale_factors) == self.n_scales
            self.scale_factors = scale_factors

        self.name = name
        self.dtype = read_param(dtype, name, str)
        self.path = read_param(path, name, str)
        self.paths_in_file = read_param(paths_in_file, name, list)
        # we assume that the slicing config is w.r.t. the volume at highest resolution
        # and that this is the volume at index 0 of `paths_in_dataset`
        # TODO add option to switch between n5, h5 etc.
        self.dataset0 = io.LazyN5VolumeLoader(self.path, self.paths_in_file[0],
                                              name=name, **slicing_config)
        # load the volumes at lower scales
        self.datasets = [z5py.File(self.path)[pid] for pid in self.paths_in_file[1:]]
        # we need offset for proper coordinate transformations.
        # we assume that these are stored as attribute `offset` in the
        # datasets
        self.offsets = [z5py.File(self.path)[pid].attrs['offset']
                        for pid in self.paths_in_file[1:]]
        self.transforms = self.get_transforms(mean, std)
        super().__init__(self.dataset0.base_sequence)

    def get_transforms(self, mean, std):
        transforms = Compose(Cast(self.dtype),
                             Normalize(mean=mean, std=std))
        return transforms

    # get the data at lower scale
    def _get_scale(self, slices, scale):
        scale_factor = self.scale_factors[scale]
        # need to substract 1 when indexing, because we store
        # the dataset for 0th scale differently and don't have offsets
        ds = self.datasets[scale - 1]
        offset = self.offsets[scale - 1]

        # downscale the bounding box defined by slices
        if isinstance(scale_factor, int):
            slices_ = tuple(slice(sl.start // scale_factor + off,
                                  sl.stop // scale_factor + off)
                            for sl, off in zip(slices, offset))
        elif isinstance(scale_factor, (tuple, list)):
            slices_ = tuple(slice(sl.start // sf + off, sl.stop // sf + off)
                            for sl, sf, off in zip(slices, scale_factor, offset))
        else:
            raise NotImplementedError

        # enlarge the bounding box according to our downscale policy
        if self.downscale_policy is not None:
            original_shape = tuple(sl.stop - sl.start for sl in slices)
            scale_shape = tuple(sl.stop - sl.start for sl in slices_)
            if isinstance(self.downscale_policy, int):
                shape_factor = self.downscale_policy
                mandatory_divisor = None
            elif isinstance(self.downscale_policy, tuple):
                shape_factor, mandatory_divisor = self.downscale_policy
            else:
                raise NotImplementedError
            target_shape = tuple(min(int(sh * shape_factor), dsh)
                                 for sh, dsh in zip(original_shape, scale_shape))

            # change the target shape to fit the mandatory divisor if given
            if mandatory_divisor is not None:
                mandatory_divisor = mandatory_divisor if isinstance(mandatory_divisor,
                                                                    (tuple, list)) else\
                    (mandatory_divisor,) * len(target_shape)
                target_shape = tuple(ts if ts % md == 0 else ts + (md - ts % md)
                                     for ts, md in zip(target_shape, mandatory_divisor))

            shape_diff = tuple((os - ts) / 2
                               for os, ts in zip(original_shape, target_shape))
            # FIXME this will not yield the correct size for odd target shapes
            slices_ = tuple(slice(sl.start - floor(sd),
                                  sl.stop + ceil(sd)) for sl, sd in zip(slices_,
                                                                        shape_diff))

        return ds[slices_]

    def __getitem__(self, index):
        # first, we load the chunk at original scale
        index = int(index)
        v0 = self.dataset0[index]

        # next, we get the slices at original scale and transform
        # them to the lower scales
        slices = tuple(self.dataset0.base_sequence[index])
        # we need to make sure that the len of slices corresponds to
        # our number of dimensions TODO don't hard-code to 3
        assert len(slices) == 3, str(len(slices))
        volumes = [v0]
        for scale in range(1, self.n_scales):
            volumes.append(self._get_scale(slices, scale))

        if self.transforms is None:
            transformed = volumes
        else:
            transformed = [self.transforms(vol) for vol in volumes]

        return transformed
