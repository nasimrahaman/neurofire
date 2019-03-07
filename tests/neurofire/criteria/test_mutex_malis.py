import unittest
import pickle
import numpy as np
import torch

try:
    import affogato
    WITH_AFF = True
except ImportError:
    WITH_AFF = False

try:
    from base_test import BaseTest
except ImportError:
    from .base_test import BaseTest


class TestMutexMalis(BaseTest):

    @unittest.skipUnless(WITH_AFF, "Need affogato")
    def test_mutex_malis(self):
        from neurofire.criteria.mutex_malis import MutexMalisLoss
        seg = torch.from_numpy(self.make_segmentation(self.shape)[None, None])
        aff_shape = (1, 6) + self.shape

        pred = torch.rand(*aff_shape, dtype=torch.float, requires_grad=True)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3]]
        criterion = MutexMalisLoss(offsets=offsets, number_of_attractive_channels=3)

        loss = criterion(pred, seg)
        self.assertNotEqual(loss.item(), 0)
        loss.backward()

        grads = pred.grad.data.numpy()
        self.assertEqual(grads.shape, pred.shape)
        self.assertFalse(np.allclose(grads, 0))

    @unittest.skipUnless(WITH_AFF, "Need affogato")
    def test_pickle(self):
        from neurofire.criteria.malis import MalisLoss
        m0 = MalisLoss(ndim=2)
        p = pickle.dumps(m0)
        m1 = pickle.loads(p)
        #
        self.assertEqual(m1.ndim, m0.ndim)
        self.assertEqual(m1.offsets, m0.offsets)


if __name__ == '__main__':
    print("BLub")
    unittest.main()
