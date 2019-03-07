import unittest
import numpy as np
import torch

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
try:
    from .base_test import BaseTest
except ImportError:
    from base_test import BaseTest

try:
    import affogato
    WITH_AFF = True
except ImportError:
    WITH_AFF = False


class TestLossTransforms(BaseTest):

    def test_mask_ignore_label(self):
        from neurofire.criteria.loss_transforms import MaskIgnoreLabel
        from neurofire.transform.segmentation import Segmentation2Membranes
        trafo = MaskIgnoreLabel(-1)
        seg_trafo = Segmentation2Membranes()

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = seg == 0

        # make membrane target
        target = seg_trafo(seg)
        target[ignore_mask] = -1
        target = torch.from_numpy(target)
        target.requires_grad = False
        self.assertEqual(target.shape, seg.shape)

        # make dummy torch prediction
        prediction = torch.Tensor(*self.shape).uniform_(0, 1)
        prediction.requires_grad = True
        masked_prediction, _ = trafo(prediction, target)

        self.assertEqual(prediction.size(), masked_prediction.size())

        # apply a loss to the prediction and check that the
        # masked parts are actually zero

        # apply cross entropy loss
        criterium = SorensenDiceLoss()
        loss = criterium(masked_prediction, target)
        loss.backward()

        grads = prediction.grad.data.numpy()
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertTrue((grads[np.logical_not(ignore_mask)] != 0).all())

    # @unittest.skipUnless(WITH_AFF, "need affogato")
    @unittest.skip
    def test_transition_mask(self):
        from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
        from neurofire.transform.affinities import Segmentation2Affinities

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9),
                   (9, 4, 0), (4, 9, 0), (9, 0, 9)]
        trafo = MaskTransitionToIgnoreLabel(offsets, ignore_label=0)
        aff_trafo = Segmentation2Affinities(offsets=offsets, retain_segmentation=True,
                                            retain_mask=False)

        seg = self.make_segmentation_with_ignore(self.shape)
        ignore_mask = self.brute_force_transition_masking(seg, offsets)
        target = torch.from_numpy(aff_trafo(seg.astype('float32'))[None])
        target.requires_grad = False

        tshape = target.size()
        pshape = (tshape[0], tshape[1] - 1) + tshape[2:]
        prediction = torch.Tensor(*pshape).uniform_(0, 1)
        prediction.require_grad = True
        masked_prediction, target_ = trafo(prediction, target)

        self.assertEqual(masked_prediction.size(), prediction.size())
        self.assertEqual(target.size(), target_.size())
        self.assertTrue(np.allclose(target.data.numpy(), target_.data.numpy()))

        # apply cross entropy loss
        criterium = SorensenDiceLoss()
        target = target[:, 1:]
        self.assertEqual(target.size(), masked_prediction.size())
        loss = criterium(masked_prediction, target)
        loss.backward()

        grads = prediction.grad.data.numpy().squeeze()
        self.assertEqual(grads.shape, ignore_mask.shape)
        self.assertTrue((grads[ignore_mask] == 0).all())
        self.assertFalse(np.sum(grads[np.logical_not(ignore_mask)]) == 0)


if __name__ == '__main__':
    unittest.main()
