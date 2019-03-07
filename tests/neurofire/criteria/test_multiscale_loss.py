import unittest
import torch
from torch.autograd import Variable
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.io.transform.base import Compose

try:
    import affogato
    WITH_AFF = True
except ImportError:
    WITH_AFF = False

try:
    from base_test import BaseTest
except ImportError:
    from .base_test import BaseTest


class TestMultiscale(BaseTest):

    @unittest.skipUnless(WITH_AFF, "need affogato")
    def test_maxpool_loss(self):
        from neurofire.criteria.loss_wrapper import LossWrapper
        from neurofire.criteria.multi_scale_loss import MultiScaleLossMaxPool
        from neurofire.transform.segmentation import Segmentation2Affinities

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9)]

        shape = (128, 128, 128)
        aff_trafo = Segmentation2Affinities(offsets, retain_segmentation=False)
        seg = self.make_segmentation_with_ignore(shape)

        target = Variable(torch.Tensor(aff_trafo(seg.astype('float32'))[None]),
                          requires_grad=False)

        tshape = target.size()
        # make all scale predictions
        predictions = []
        for scale in range(4):
            pshape = tuple(tshape[:2],) + shape
            predictions.append(Variable(torch.Tensor(*pshape).uniform_(0, 1),
                                        requires_grad=True))
            shape = tuple(sh // 2 for sh in shape)

        criterion = LossWrapper(SorensenDiceLoss())
        ms_loss = MultiScaleLossMaxPool(criterion, 2)
        loss = ms_loss.forward(predictions, target)
        loss.backward()

        for prediction in predictions:
            grads = prediction.grad.data
            # check for the correct gradient size
            self.assertEqual(grads.size(), prediction.size())
            # check that gradients are not trivial
            self.assertNotEqual(grads.sum(), 0)

    def _test_maxpool_loss_retain_segmentation(self):
        from neurofire.criteria.loss_wrapper import LossWrapper
        from neurofire.criteria.multi_scale_loss import MultiScaleLossMaxPool
        from neurofire.transform.segmentation import Segmentation2AffinitiesFromOffsets
        from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
        from neurofire.criteria.loss_transforms import RemoveSegmentationFromTarget

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (9, 0, 0), (0, 9, 0), (0, 0, 9)]

        shape = (128, 128, 128)
        aff_trafo = Segmentation2AffinitiesFromOffsets(3, offsets, retain_segmentation=True,
                                                       add_singleton_channel_dimension=True)
        seg = self.make_segmentation_with_ignore(shape)

        target = Variable(torch.Tensor(aff_trafo(seg.astype('float32'))[None]),
                          requires_grad=False)

        tshape = target.size()
        # make all scale predictions
        predictions = []
        for scale in range(4):
            pshape = (tshape[0], tshape[1] - 1) + shape
            predictions.append(Variable(torch.Tensor(*pshape).uniform_(0, 1),
                                        requires_grad=True))
            shape = tuple(sh // 2 for sh in shape)

        trafos = Compose(MaskTransitionToIgnoreLabel(offsets, ignore_label=0),
                         RemoveSegmentationFromTarget())
        criterion = LossWrapper(SorensenDiceLoss(), trafos)
        ms_loss = MultiScaleLossMaxPool(criterion, 2, retain_segmentation=True)
        loss = ms_loss.forward(predictions, target)
        loss.backward()

        for prediction in predictions:
            grads = prediction.grad.data
            # check for the correct gradient size
            self.assertEqual(grads.size(), prediction.size())
            # check that gradients are not trivial
            self.assertNotEqual(grads.sum(), 0)


if __name__ == '__main__':
    unittest.main()
