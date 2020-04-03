from functools import partial
from concurrent import futures

import numpy as np
import torch

# TODO remove all vigra dependencies
try:
    import vigra
except ImportError:
    pass

from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import label

import inferno.utils.python_utils as pyu
from inferno.extensions.metrics.arand import ArandError

try:
    from affogato.segmentation import connected_components, compute_mws_segmentation
    HAVE_AFFOGATO = True
except ImportError:
    HAVE_AFFOGATO = False

try:
    import nifty
    import nifty.graph.rag as nrag
    import nifty.graph.opt.multicut as nmc
    HAVE_NIFTY = True
except ImportError:
    HAVE_NIFTY = False


class ArandFromSegmentationBase(ArandError):
    """
    Base class for segmentation based arand error.
    Subclasses must implement `input_to_segmentation`.

    Arguments:
        parameters [list]: list of different parameters for segmentation algorithm
            that should be evaluated. only single type of parameter supported for now. (default: None)
        average_slices [bool]: evaluate the score as average over 2d slices (default: True)
    """

    def __init__(self, parameters=None, average_slices=False):
        super().__init__(average_slices=average_slices)
        self.parameters = parameters if parameters is None else pyu.to_iterable(parameters)

    def input_to_segmentation(self, input_batch, parameter):
        raise NotImplementedError("Implement `input_to_segmentation` in subclass")

    def forward(self, prediction, target):
        assert prediction.dim() in [4, 5], \
            "`prediction` must be a 4D batch of 2D images or a 5D batch of 3D volumes."

        input_batch = prediction.cpu().numpy()
        if(np.isnan(input_batch).any()):
            raise RuntimeError("Have nans in prediction")
        gt_seg = target[:, 0:1]

        if self.parameters is None:
            seg = self.input_to_segmentation(input_batch)
            return super().forward(seg, gt_seg)
        else:
            arand_errors = []
            # iterate over the different parameters we have
            for param in self.parameters:
                # compute the mean arand erros for all batches for the given threshold
                seg = self.input_to_segmentation(input_batch, param)
                arand_errors.append(super().forward(seg, gt_seg))
                # return the best arand errror
                return min(arand_errors)


class ArandErrorFromConnectedComponentsOnAffinities(ArandFromSegmentationBase):
    def __init__(self, thresholds=0.5, invert_affinities=False,
                 normalize_affinities=False, **super_kwargs):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        super().__init__(thresholds, **super_kwargs)
        self.invert_affinities = invert_affinities
        self.normalize_affinities = normalize_affinities

    def input_to_segmentation(self, input_batch, thresh):
        dim = input_batch.ndim - 2
        # if specified, invert and / or normalize the affinities
        if self.invert_affinities:
            input_batch = 1. - input_batch
        if self.normalize_affinities:
            input_batch = input_batch / input_batch.max()
        # Compute the segmentation via connected components on the affinities
        ccs = np.array([connected_components(batch[:dim], thresh)[0] for batch in input_batch])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(ccs[:, None].astype('int32'))


class ArandErrorFromConnectedComponents(ArandFromSegmentationBase):
    def __init__(self, thresholds=0.5, invert_input=False,
                 average_input=False, normalize_input=False, **super_kwargs):
        super().__init__(thresholds, **super_kwargs)
        self.invert_input = invert_input
        self.normalize_input = normalize_input
        self.average_input = average_input

    # Compute the segmentation via connected components on the input
    def input_to_segmentation(self, input_batch, thresh):
        dim = input_batch.ndim - 2

        # if specified, invert and / or normalize the input
        if self.invert_input:
            input_batch = 1. - input_batch
        if self.normalize_input:
            input_batch -= input_batch.min()
            input_batch = input_batch / input_batch.max()

        if input_batch.shape[1] > 1:
            assert self.average_input, "Need to allow average input for multi-channel predictions"
            # we only average over the nearest neighbor affinities
            thresholded = (np.mean(input_batch[:, :dim], axis=1) >= thresh).astype('uint8')
        else:
            thresholded = (input_batch[:, 0] >= thresh).astype('uint8')
        ccs = np.array([label(threshd) for threshd in thresholded])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(ccs[:, None].astype('int32'))


class ArandErrorFromMWS(ArandFromSegmentationBase):
    def __init__(self, offsets, strides=None, randomize_strides=False,
                 **super_kwargs):
        assert HAVE_AFFOGATO, "Couldn't find 'affogato' module, affinity calculation is not available"
        super().__init__(**super_kwargs)
        self.offsets = offsets
        self.dim = len(offsets[0])
        self.strides = strides
        self.randomize_strides = randomize_strides

    def _run_mws(self, input_):
        assert len(input_) == len(self.offsets)
        input_[:self.dim] *= -1
        input_[:self.dim] += 1
        return compute_mws_segmentation(input_, self.offsets,
                                        number_of_attractive_channels=self.dim,
                                        strides=self.strides,
                                        randomize_strides=self.randomize_strides)

    def input_to_segmentation(self, input_batch):
        dim = input_batch.ndim - 2
        assert dim == self.dim
        seg = np.array([self._run_mws(batch) for batch in input_batch])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(seg[:, None].astype('int32'))


class ArandErrorFromMulticut(ArandFromSegmentationBase):
    def __init__(self, betas=.5, offsets=None,
                 dt_threshold=.25, dt_sigma=2., use_2d_ws=False,
                 size_filter=25, weight_edges=False, n_threads=8,
                 **super_kwargs):
        assert HAVE_NIFTY, "Need nifty to run multicut validation"
        super().__init__(betas, **super_kwargs)
        self.offsets = offsets
        self.dt_threshold = dt_threshold
        self.dt_sigma = dt_sigma
        self.use_2d_ws = use_2d_ws
        self.size_filter = size_filter
        self.weight_edges = weight_edges
        self.n_threads = n_threads

    @staticmethod
    def _filter_by_size(input_, seg, size_filter):
        ids, sizes = np.unique(seg, return_counts=True)
        mask = np.in1d(seg, ids[sizes < size_filter]).reshape(seg.shape)
        seg[mask] = 0
        vigra.analysis.watershedsNew(input_, seeds=seg, out=seg)
        seg, max_id, _ = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)
        return seg, max_id + 1

    # watershed on distance transform
    def _compute_ws_impl(self, input_):
        thresholded = input_ < self.dt_threshold
        dt = distance_transform_edt(thresholded).astype('float32')
        if self.dt_sigma > 0.:
            dt = vigra.filters.gaussianSmoothing(dt, self.dt_sigma)
        if input_.ndim == 2:
            seeds = vigra.analysis.localMaxima(dt, allowPlateaus=True,
                                               allowAtBorder=True, marker=np.nan)
        else:
            seeds = vigra.analysis.localMaxima3D(dt, allowPlateaus=True,
                                                 allowAtBorder=True, marker=np.nan)
        seeds = vigra.analysis.labelMultiArrayWithBackground(np.isnan(seeds).view('uint8'))
        ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
        if self.size_filter > 0:
            ws, max_id = self._filter_by_size(input_, ws, self.size_filter)
        return ws, max_id + 1

    def _compute_ws(self, input_):
        # for 4d input we need to agglomerate channels first
        # here, we take the max over the first 3 channels (full 3d case) /
        # max over channel 1 and 2 (use_2d_ws == True)
        if input_.ndim == 4:
            hmap = np.max(input_[1:3], axis=0) if self.use_2d_ws else np.max(input_[:3], axis=0)
        else:
            hmap = input_

        # check if we run watershed in 2d and stack
        # or if we run purely in 3d
        if self.use_2d_ws:
            # compute watersheds for individual slices in parallel
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                tasks = [tp.submit(self._compute_ws_impl, hmap[z]) for z in range(hmap.shape[0])]
                res = [t.result() for t in tasks]
            seg = np.concatenate([r[0][None] for r in res], axis=0)
            # compute the offsets that need to be added to each slice to
            # make the labels consecutive
            offsets = np.array([r[1] for r in res], dtype='uint32')
            last_max_id = offsets[-1]
            offsets = np.cumsum(np.roll(offsets, 1))
            # add offsets to the segmentation
            seg += offsets[:, None, None]
            n_labels = offsets[-1] + last_max_id
            return seg, n_labels
        else:
            return self._compute_ws_impl(hmap)

    # edge probabilities to multicut costs
    def _probs_to_costs(self, probs, edge_len, beta):
        p_min = 0.001
        p_max = 1. - p_min
        costs = (p_max - p_min) * probs + p_min
        costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
        # weight the costs with edge len
        if self.weight_edges:
            w = edge_len / edge_len.max()
            costs *= w
        return costs

    def _compute_mc(self, input_, feat_function, beta):
        # watershed and region adjacency graph
        ws, n_labels = self._compute_ws(input_)

        rag = nrag.gridRag(ws, numberOfLabels=n_labels,
                           numberOfThreads=self.n_threads)
        if rag.numberOfEdges == 0:
            return np.zeros_like(ws)

        # features and features to costs
        feats = feat_function(rag, input_)
        probs, edge_len = feats[:, 0], feats[:, -1]
        costs = self._probs_to_costs(probs, edge_len, beta)

        # graph and multicut solver
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        objective = nmc.multicutObjective(graph, costs)
        solver = objective.kernighanLinFactory(warmStartGreedy=True).create(objective)

        # solve multicut and project back to segmentation
        # TODO time limit
        node_labels = solver.optimize()
        return nrag.projectScalarNodeDataToPixels(rag, node_labels,
                                                  numberOfThreads=self.n_threads)

    # Compute the segmentation via multicut on the input
    def input_to_segmentation(self, input_batch, beta):
        dim = input_batch.ndim - 2

        n_channels = input_batch.shape[1]
        # more than one channel -> affinity input
        if n_channels > 1:
            assert self.offsets is not None, "Need to have offsets for affinity predictions"
            assert len(self.offsets) <= n_channels
            assert dim == len(self.offsets[0])
            feat_function = partial(nrag.accumulateAffinityStandartFeatures,
                                    offsets=self.offsets,
                                    numberOfThreads=self.n_threads)
        # only one channel -> boundary map input
        else:
            # FIXME this does not return the edge lens in last axis
            feat_function = partial(nrag.accumulateEdgeStandartFeatures,
                                    numberOfThreads=self.n_threads)

        segs = np.array([self._compute_mc(inp, feat_function, beta) for inp in input_batch])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(segs[:, None].astype('int32'))
