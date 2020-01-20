from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import Cast
from inferno.utils import python_utils as pyu
from ....transform.segmentation import Segmentation2Membranes
from ....transform.segmentation import NegativeExponentialDistanceTransform
from ....transform.segmentation import ConnectedComponents3D, ConnectedComponents2D


class MembraneVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None,
                 data_slice=None, name=None, dtype='float32', nedt_gain=None,
                 **slicing_config):
        path_in_h5_dataset = path_in_h5_dataset if path_in_h5_dataset is not None else \
            '/volumes/labels/neuron_ids'
        # Init super
        super(MembraneVolume, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                             data_slice=data_slice, name=name, **slicing_config)
        # Validate and record attributes
        assert isinstance(dtype, str)
        assert isinstance(nedt_gain, (type(None), int, float))
        self.dtype = dtype
        self.nedt_gain = nedt_gain or 1.
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(Segmentation2Membranes(dtype=self.dtype),
                             NegativeExponentialDistanceTransform(gain=self.nedt_gain),
                             Cast(self.dtype))
        return transforms


class AffinityVolume(MembraneVolume):
    def __init__(self, path, path_in_h5_dataset=None, data_slice=None, name=None,
                 dtype='float32', affinity_dim=3, affinity_order=1, retain_segmentation=False,
                 **slicing_config):
        # Set attributes
        self.affinity_dim = affinity_dim
        self.affinity_order = affinity_order
        self.retain_segmentation = retain_segmentation
        # Initialize super
        super(AffinityVolume, self).__init__(path, path_in_h5_dataset=path_in_h5_dataset,
                                             data_slice=data_slice, name=name,
                                             dtype=dtype, nedt_gain=None,
                                             **slicing_config)
        # We don't need a nedt_gain anymore
        del self.nedt_gain

    def get_transforms(self):
        from ....transform.segmentation import Segmentation2MultiOrderAffinities
        # The Segmentation2Affinities adds a channel dimension. Now depending on how many
        # orders were requested, we dispatch Segmentation2Affinities or
        # Segmentation2MultiOrderAffinities.
        transforms = Compose()
        # Cast to the right dtype
        transforms.add(Cast(self.dtype))
        # Run connected components to shuffle the labels
        transforms.add(ConnectedComponents3D())
        # Make affinity maps
        transforms.add(
            Segmentation2MultiOrderAffinities(dim=self.affinity_dim,
                                              orders=pyu.to_iterable(self.affinity_order),
                                              add_singleton_channel_dimension=True,
                                              retain_segmentation=self.retain_segmentation))
        return transforms


class SegmentationVolume(HDF5VolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None,
                 data_slice=None, name=None, dtype='float32', apply_on_image=False,
                 **slicing_config):
        path_in_h5_dataset = path_in_h5_dataset if path_in_h5_dataset is not None else \
            '/volumes/labels/neuron_ids'
        # Init super
        super(SegmentationVolume, self).__init__(path=path, path_in_h5_dataset=path_in_h5_dataset,
                                                 data_slice=data_slice, name=name, **slicing_config)

        assert isinstance(dtype, str)
        self.dtype = dtype
        self.apply_on_image = apply_on_image
        # Make transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        if self.apply_on_image:
            transforms = Compose(ConnectedComponents2D(), Cast(self.dtype))
        else:
            transforms = Compose(ConnectedComponents3D(), Cast(self.dtype))
        return transforms
