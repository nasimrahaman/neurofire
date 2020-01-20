import numpy as np

from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.core import ZipReject
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast, Normalize, AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform


class RejectNonZeroThreshold(object):
    def __init__(self, threshold):
        self.threshold = threshold

    # return True if ration of non-zeros is below
    # the threshold and hence the batch should be rejected
    def __call__(self, fetched):
        return (np.count_nonzero(fetched) / fetched.size) < self.threshold


class ArtifactVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None, data_slice=None,
                 dtype='float32', **slicing_config):

        # Initialize super with path to gt_cleaned.h5
        super().__init__(path, path_in_h5_dataset=path_in_h5_dataset,
                         data_slice=data_slice,
                         **slicing_config)
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(Cast(self.dtype))
        return transforms


class ArtifactSource(ZipReject):

    def __init__(self, volume_config, slicing_config, min_masking_ratio, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'artifacts' in volume_config
        assert 'alpha_mask' in volume_config

        # get artifact and alpha mask volumes
        artifact_volume_kwargs = dict(volume_config.get('artifacts'))
        artifact_volume_kwargs.update(slicing_config)
        self.artifact_volume = ArtifactVolume(**artifact_volume_kwargs)

        alpha_mask_volume_kwargs = dict(volume_config.get('alpha_mask'))
        alpha_mask_volume_kwargs.update(slicing_config)
        self.alpha_mask_volume = HDF5VolumeLoader(**alpha_mask_volume_kwargs)

        # initialize the rejection function,
        # which rejects all slices which have nonzero alpha mask
        # smaller than min_masking_ratio
        rejecter = RejectNonZeroThreshold(min_masking_ratio)

        # Initialize ZipReject
        super().__init__(self.artifact_volume, self.alpha_mask_volume,
                         rejection_dataset_indices=1,
                         rejection_criterion=rejecter, sync=True)
        # Set master config
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        all_transforms = [RandomRotate()]
        if 'elastic_transform' in self.master_config:
            all_transforms.append(ElasticTransform(**self.master_config.get('elastic_transform',
                                                                            {})))
        if self.master_config.get('crop_after_elastic_transform', False):
            all_transforms\
                .append(CenterCrop(**self.master_config.get('crop_after_elastic_transform')))
        all_transforms.append(AsTorchBatch(2))
        transforms = Compose(*all_transforms)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        min_masking_ratio = config.get('min_masking_ratio')
        return cls(volume_config=volume_config, slicing_config=slicing_config,
                   master_config=master_config, min_masking_ratio=min_masking_ratio)
