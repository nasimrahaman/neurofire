from inferno.io.core import Zip
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from neurofire.datasets.loader import RawVolume, SegmentationVolume
from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets


class ISBI2012Dataset3D(Zip):
    def __init__(self, volume_config, slicing_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)

        # Build raw volume
        self.raw_volume = RawVolume(**raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(**segmentation_volume_kwargs)

        super().__init__(self.raw_volume, self.segmentation_volume, sync=True)
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate())

        # Elastic transforms can be skipped by
        # setting elastic_transform to false in the
        # yaml config file.
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                            sigma=elastic_transform_config.get('sigma', 50.),
                                            order=elastic_transform_config.get('order', 0)))

        # affinity transforms for affinity targets
        # we apply the affinity target calculation only to the segmentation (1)
        assert self.affinity_config is not None
        aff_trafo = Segmentation2AffinitiesFromOffsets(apply_to=[1], **self.affinity_config)
        transforms.add(aff_trafo)

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        # make 3d torch batch
        transforms.add(AsTorchBatch(3))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(volume_config=volume_config,
                   slicing_config=slicing_config,
                   master_config=master_config)


def get_isbi_loader_3d(config):
    """
    Gets Isbi loader given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    loader_config = config.pop('loader_config')
    dataset = ISBI2012Dataset3D.from_config(config)
    loader = DataLoader(dataset, **loader_config)
    return loader
