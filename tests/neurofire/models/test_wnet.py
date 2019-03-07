import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester


class TestWnet(unittest.TestCase):
    def test_wnet_2d(self):
        from neurofire.models import WNet2D
        tester = ModelTester((1, 1, 512, 512), (1, 1, 512, 512))
        if cuda.is_available():
            tester.cuda()
        tester(WNet2D(1, 1,
                      initial_num_fmaps=12,
                      fmap_growth=3))


if __name__ == '__main__':
    unittest.main()
