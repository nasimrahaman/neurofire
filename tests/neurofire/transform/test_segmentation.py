import unittest
import numpy as np


class TestSegmentation(unittest.TestCase):

    # generate a random segmentation
    @staticmethod
    def generate_segmentation(shape):
        segmentation = np.zeros(shape, dtype='uint32')
        label = 0
        for index in range(segmentation.size):
            coord = np.unravel_index(index, shape)
            segmentation[coord] = label
            if np.random.random() > .9:  # change label with 10% probability
                label += 1
        return segmentation

    # generate a toy segmentation
    @staticmethod
    def generate_toy_data():
        seg = np.zeros((10, 10), dtype='uint32')
        seg[:5, :5] = 1
        seg[:5, 5:] = 2
        seg[5:, :5] = 3
        seg[5:, 5:] = 4

        aff = np.ones((2, 10, 10), dtype='float32')
        aff[0, 5, :] = 0
        aff[1, :, 5] = 0

        # these are the undefined affinities, gonna set them to 0.5 for now
        aff[0, 0, :] = 0.
        aff[1, :, 0] = 0.
        return seg, aff

    @staticmethod
    def affinities_brute_force(segmentation, offsets):
        shape = segmentation.shape
        affinities = np.zeros((len(offsets),) + shape, dtype='float32')

        # iterate over all edges (== 1d affinity coordinates) to get the correct affinities
        for edge in range(affinities.size):
            aff_coord = np.unravel_index(edge, affinities.shape)
            offset = offsets[aff_coord[0]]
            coord_u = aff_coord[1:]
            coord_v = tuple(cu + off for cu, off in zip(coord_u, offset))

            # check if coord_v is valid
            if any(cv < 0 or cv >= sha for cv, sha in zip(coord_v, shape)):
                # affinities[aff_coord] = 1.
                continue

            # write the correct affinity (0 -> disconnected, 1 -> connected)
            u, v = segmentation[coord_u], segmentation[coord_v]
            affinities[aff_coord] = 1. if u == v else 0.

        return affinities

    def test_seg2mem(self):
        from neurofire.transform.segmentation import Segmentation2Membranes
        trafo = Segmentation2Membranes()
        for dim in (2, 3):
            shape = dim * (128,)
            seg = self.generate_segmentation(shape)
            membranes = trafo(seg)
            self.assertEqual(membranes.shape, shape)
            # make for torch tensor, check that agree
            # membranes_torch =
            # self.assertEqual(membranes_torch.shape, shape)

    def test_brute_force_affs_toy(self):
        offsets = [(-1, 0), (0, -1)]
        segmentation, expected = self.generate_toy_data()
        output = self.affinities_brute_force(segmentation, offsets)
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(np.allclose(output, expected))

    def test_affs_toy(self):
        from neurofire.transform.affinities import Segmentation2Affinities
        segmentation, expected = self.generate_toy_data()
        offsets = [[-1, 0], [0, -1]]
        transform = Segmentation2Affinities(offsets=offsets)
        output = transform(segmentation)
        # print(output)
        # print()
        # print(expected)
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(np.allclose(output, expected))

    def test_affs_random(self):
        from neurofire.transform.affinities import Segmentation2Affinities
        offsets_2d = [[-1, 0], [0, -1]]
        offsets_3d = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

        # 2D with 2D affinities
        wannabe_groundtruth = np.random.uniform(size=(512, 512))
        # Build transform
        transform = Segmentation2Affinities(offsets=offsets_2d)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 2)

        # 3D with 3D affinities
        wannabe_groundtruth = np.random.uniform(size=(16, 512, 512))
        # Build transform
        transform = Segmentation2Affinities(offsets=offsets_3d)
        output = transform(wannabe_groundtruth)
        self.assertSequenceEqual((16, 512, 512), output.shape[1:])
        self.assertEqual(output.shape[0], 3)

    def test_affs_from_offsets_2d(self):
        from neurofire.transform.affinities import Segmentation2Affinities
        shape = (64, 64)
        segmentation = self.generate_segmentation(shape).astype('int32')

        offsets = [(1, 0), (0, 1),
                   (3, 0), (0, 3),
                   (1, 1), (-1, 1)]

        trafo = Segmentation2Affinities(offsets=offsets)
        output = trafo(segmentation)
        output_expected = self.affinities_brute_force(segmentation.squeeze(), offsets)
        self.assertEqual(output.shape, output_expected.shape)
        # print(np.sum(np.isclose(output, output_expected)), '/', output.size)
        self.assertTrue(np.allclose(output, output_expected))

    def test_affs_from_offsets_3d(self):
        from neurofire.transform.affinities import Segmentation2Affinities
        shape = (64, 64, 64)
        segmentation = self.generate_segmentation(shape).astype('int32')

        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (3, 0, 0), (0, 3, 0), (0, 0, 3),
                   (1, 1, 0), (-1, 1, 0)]

        trafo = Segmentation2Affinities(offsets=offsets)
        output = trafo(segmentation)
        output_expected = self.affinities_brute_force(segmentation.squeeze(), offsets)
        self.assertEqual(output.shape, output_expected.shape)
        # print(np.sum(np.isclose(output, output_expected)), '/', output.size)
        self.assertTrue(np.allclose(output, output_expected))

    def test_cc_2d(self):
        from neurofire.transform.segmentation import ConnectedComponents2D
        x = np.array([[1, 1, 0, 1, 1],
                      [1, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0],
                      [1, 0, 1, 1, 0],
                      [0, 1, 1, 1, 1]],
                     dtype='uint32')
        for label_segmentation in (False, True):
            cc = ConnectedComponents2D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3])

    def test_cc_3d(self):
        from neurofire.transform.segmentation import ConnectedComponents3D
        x = np.array([[[1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]],
                      [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]],
                      [[1, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]]],
                     dtype='uint32')
        for label_segmentation in (False, True):
            cc = ConnectedComponents3D(label_segmentation=label_segmentation)
            y = cc(x)
            self.assertEqual(y.shape, x.shape)
            uniques = np.unique(y).tolist()
            self.assertEqual(uniques, [0, 1, 2, 3, 4])

    def test_cc_label_segmentation(self):
        from neurofire.transform.segmentation import ConnectedComponents2D
        x = np.array([[1, 2, 1],
                      [1, 1, 1],
                      [1, 2, 1]],
                     dtype='uint32')
        cc = ConnectedComponents2D(label_segmentation=True)
        y = cc(x)
        self.assertEqual(y.shape, x.shape)
        uniques = np.unique(y).tolist()
        self.assertEqual(uniques, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
