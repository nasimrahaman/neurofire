from inferno.io.transform import Compose, Transform
from inferno.io.transform.generic import Cast
import inferno.io.volumetric as io

from neurofire.transform.segmentation import ConnectedComponents3D


class BinarizeSegmentation(Transform):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def volume_function(self, input_):
        return (input_ > 0).astype(input_.dtype)


class SegmentationVolume(io.HDF5VolumeLoader):
    def __init__(self, path, path_in_file,
                 data_slice=None, name=None, dtype='float32',
                 label_volume=True, binarize=False, **slicing_config):
        # Init super
        super(SegmentationVolume, self).__init__(path=path, path_in_h5_dataset=path_in_file,
                                                 data_slice=data_slice, name=name, **slicing_config)

        assert isinstance(dtype, str)
        self.dtype = dtype
        self.label_volume = label_volume
        self.binarize = binarize
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = []
        if self.label_volume:
            transforms.append(ConnectedComponents3D())
        if self.binarize:
            transforms.append(BinarizeSegmentation())
        transforms.append(Cast(self.dtype))
        return Compose(*transforms)


class N5SegmentationVolume(io.LazyN5VolumeLoader):
    def __init__(self, path, path_in_file,
                 data_slice=None, name=None, dtype='float32',
                 label_volume=True, **slicing_config):
        # Init super
        super(N5SegmentationVolume, self).__init__(path=path, path_in_file=path_in_file,
                                                   data_slice=data_slice, name=name, **slicing_config)

        assert isinstance(dtype, str)
        self.label_volume = label_volume
        self.dtype = dtype
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        if self.label_volume:
            transforms = Compose(ConnectedComponents3D(),
                                 Cast(self.dtype))
        else:
            transforms = Cast(self.dtype)
        return transforms
