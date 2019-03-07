import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import MultiscaleModelTester


class TestM2FCN(unittest.TestCase):
    def test_m2fcn_2d(self):
        from neurofire.models import M2FCN
        shape = (1, 1, 64, 64)
        tester = MultiscaleModelTester(shape, 18 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(M2FCN(1, 1,  3 * (16,), block_type_key='default'))

    def test_m2fcn_3d(self):
        from neurofire.models import M2FCN
        shape = (1, 1, 64, 64, 64)
        tester = MultiscaleModelTester(shape, 18 * [shape])
        if cuda.is_available():
            tester.cuda()
        tester(M2FCN(1, 1,  3 * (16,), block_type_key='default3d',
                     output_type_key='default3d',
                     sampling_type_key='default3d'))
        tester(M2FCN(1, 1,  3 * (16,), block_type_key='default3d',
                     output_type_key='default3d',
                     sampling_type_key='default3d'))

    # this may fail on travis due to insufficient ram
    @unittest.expectedFailure
    def test_m2fcn_3d_aniso(self):
        from neurofire.models import M2FCN
        shape = (1, 1, 32, 144, 144)
        tester = MultiscaleModelTester(shape, 18 * [shape])
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(M2FCN(1, 1, 3 * (16,),
                     scale_factor=[3, 3, 2, 2],
                     block_type_key='default3d',
                     output_type_key='default3d',
                     sampling_type_key=['anisotropic', 'anisotropic', 'default3d', 'default3d']))


if __name__ == '__main__':
    unittest.main()
