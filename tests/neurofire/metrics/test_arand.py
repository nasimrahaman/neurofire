import torch
import numpy as np
import unittest

try:
    import affogato
    WITH_AFF = True
except ImportError:
    WITH_AFF = False


class TestArand(unittest.TestCase):
    def build_input(self):
        affinity_image = np.array([[0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0]]).astype('float32')
        groundtruth_image = np.array([[1, 1, 0, 2, 2],
                                      [1, 1, 0, 2, 2],
                                      [0, 0, 0, 0, 0],
                                      [4, 4, 0, 3, 3],
                                      [4, 4, 0, 3, 3]]).astype('float32')
        affinity_batch = np.array([affinity_image for _ in range(2)])[None, ...]
        groundtruth_batch = groundtruth_image[None, None, ...]
        affinity_batch_tensor = torch.from_numpy(affinity_batch)
        groundtruth_batch_tensor = torch.from_numpy(groundtruth_batch)
        return affinity_batch_tensor, groundtruth_batch_tensor

    @unittest.skipUnless(WITH_AFF, "Need affogato for unittest on affinities")
    def test_arand_error_from_connected_components_on_affinities(self):
        from neurofire.metrics.arand import ArandErrorFromConnectedComponentsOnAffinities
        affinity, groundtruth = self.build_input()
        arand = ArandErrorFromConnectedComponentsOnAffinities(thresholds=[0.5, 0.8],
                                                              invert_affinities=True)
        error = arand(affinity, groundtruth)
        self.assertEqual(error, 0.)

    def test_arand_error_from_connected_components(self):
        from neurofire.metrics.arand import ArandErrorFromConnectedComponents
        input_, groundtruth = self.build_input()
        input_ = input_[:, 0:1]
        arand = ArandErrorFromConnectedComponents(thresholds=[0.5, 0.8],
                                                  invert_input=True)
        error = arand(input_, groundtruth)
        self.assertEqual(error, 0.)

    # TODO implement proper test, but need either 3d input or accept 2d data
    def _test_arand_error_from_multicut(self):
        from neurofire.metrics.arand import ArandErrorFromMulticut
        input_, groundtruth = self.build_input()
        arand = ArandErrorFromMulticut(dim=2)
        error = arand(input_, groundtruth)
        self.assertEqual(error, 0.)


def visualize_cc():
    import h5py
    import vigra
    from cremi_tools.viewer.volumina import view
    thresh = .9
    with h5py.File('/home/cpape/Work/data/isbi2012/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5') as f:
        affs = f['data'][:, :5, :256, :256]
    print(affs.shape)
    affs = 1. - affs
    thresholded = (np.mean(affs, axis=0) >= thresh).astype('uint8')
    # cs = vigra.analysis.labelMultiArrayWithBackground(thresholded)
    cs = vigra.analysis.labelMultiArray(thresholded)
    view([affs.transpose((1, 2, 3, 0)), thresholded, cs])


def visualize_mc():
    import h5py
    from cremi_tools.viewer.volumina import view
    from neurofire.metrics.arand import ArandErrorFromMulticut
    with h5py.File('/home/cpape/Work/data/isbi2012/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5') as f:
        affs = f['data'][:, :10, :256, :256]
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    use_2d_ws = True
    metrics = ArandErrorFromMulticut(use_2d_ws=use_2d_ws, offsets=offsets)
    beta = .5
    mc_seg = metrics.input_to_segmentation(affs[None], beta).numpy().squeeze()
    assert mc_seg.shape == affs.shape[1:]
    view([affs.transpose((1, 2, 3, 0)), mc_seg])


if __name__ == '__main__':
    # visualize_mc()
    # visualize_cc()
    unittest.main()
