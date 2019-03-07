import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestHed(unittest.TestCase):
    def test_hed_2d(self):
        from neurofire.models import HED
        shape = (1, 1, 128, 128)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(HED(1, 1, 64, 2))

    def test_hed_3d(self):
        from neurofire.models import HED
        shape = (1, 1, 32, 32, 32)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(HED(1, 1, 16, 2,
                    block_type_key='default3d',
                    output_type_key='default3d',
                    sampling_type_key='default3d'))

    # test hed 3d with anisotropic sampling
    def test_hed_3d_aniso(self):
        from neurofire.models import HED
        shape = (1, 1, 16, 64, 64)
        tester = MultiscaleModelTester(shape, 6 * [shape])
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(HED(1, 1, 16, 2,
                    block_type_key='default3d',
                    output_type_key='default3d',
                    sampling_type_key='anisotropic'))

    def _test_fusion_2d(self):
        from neurofire.models import FusionHED
        shape = (1, 1, 128, 128)
        tester = MultiscaleModelTester(shape, 19 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(FusionHED(1, 1))  # , conv_type_key='same'))

    # FIXME does not work
    def _test_fusion_3d(self):
        from neurofire.models import FusionHED
        shape = (1, 1, 64, 64, 64)
        tester = MultiscaleModelTester(shape, 19 * [shape])
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(FusionHED(1, 1,
                         conv_type_key='default3d',
                         block_type_key='default3d',
                         output_type_key='default3d',
                         upsampling_type_key='default3d'))


if __name__ == '__main__':
    unittest.main()
