import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester


class TestUnet(unittest.TestCase):
    def test_unet_2d(self):
        from neurofire.models import UNet2D
        tester = ModelTester((1, 1, 128, 128), (1, 1, 128, 128))
        if cuda.is_available():
            tester.cuda()
        tester(UNet2D(1, 1,
                      initial_num_fmaps=12,
                      fmap_growth=3))

    def test_unet_2p5d(self):
        from neurofire.models import UNet2p5D
        tester = ModelTester((1, 1, 3, 128, 128), (1, 1, 3, 128, 128))
        if cuda.is_available():
            tester.cuda()
        tester(UNet2p5D(1, 1,
                        z_channels=3,
                        initial_num_fmaps=12,
                        fmap_growth=3))

    def test_unet_3d(self):
        from neurofire.models import UNet3D
        tester = ModelTester((1, 1, 16, 128, 128), (1, 1, 16, 128, 128))
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(UNet3D(1, 1,
                      initial_num_fmaps=12,
                      fmap_growth=2,
                      scale_factor=2))
        # test unet 3d with anisotropic sampling
        tester(UNet3D(1, 1,
                      initial_num_fmaps=12,
                      fmap_growth=2,
                      scale_factor=[(1, 2, 2),
                                    (1, 2, 2),
                                    (1, 2, 2)]))


if __name__ == '__main__':
    unittest.main()
