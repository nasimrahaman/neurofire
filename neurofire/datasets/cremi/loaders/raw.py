from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast, Normalize


class RawVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None,
                 data_slice=None, name=None, dtype='float32',
                 mean=None, std=None, **slicing_config):
        path_in_h5_dataset = path_in_h5_dataset if path_in_h5_dataset is not None else \
            '/volumes/raw'
        # Init super
        super(RawVolume, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                        data_slice=data_slice, name=name, **slicing_config)
        # Record attributes
        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms(mean, std)

    def get_transforms(self, mean, std):
        transforms = Compose(Normalize(mean=mean, std=std),
                             Cast(self.dtype))
        return transforms
