from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast
import inferno.io.volumetric as io

from neurofire.transform.segmentation import ConnectedComponents3D


class SegmentationVolume(io.HDF5VolumeLoader):
    def __init__(self, path, path_in_file,
                 data_slice=None, name=None, dtype='float32',
                 preserved_label=None,
                 label_components=True, **slicing_config):
        # Init super
        super(SegmentationVolume, self).__init__(path=path, path_in_h5_dataset=path_in_file,
                                                 data_slice=data_slice, name=name, **slicing_config)

        assert isinstance(dtype, str)
        self.dtype = dtype
        self.preserved_label = preserved_label
        # Make transforms
        self.transforms = self.get_transforms(label_components)

    def get_transforms(self, label_components):
        transforms = Compose(ConnectedComponents3D(preserved_label=self.preserved_label,
                                                       label_components=label_components),
                                 Cast(self.dtype))
        return transforms


class N5SegmentationVolume(io.LazyN5VolumeLoader):
    def __init__(self, path, path_in_file,
                 data_slice=None, name=None, dtype='float32',
                 **slicing_config):
        # Init super
        super(N5SegmentationVolume, self).__init__(path=path, path_in_file=path_in_file,
                                                   data_slice=data_slice, name=name, **slicing_config)

        assert isinstance(dtype, str)
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(ConnectedComponents3D(label_segmentation=True),
                             Cast(self.dtype))
        return transforms
