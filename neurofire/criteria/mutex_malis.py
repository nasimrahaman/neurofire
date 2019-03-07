import numpy as np
import torch
from torch.autograd import Function

try:
    import affogato.learning as affl
    HAVE_AFFOGATO = True
except ImportError:
    HAVE_AFFOGATO = False


class MutexMalisLoss(Function):
    """
    Compute the constrained malis loss.
    """
    def __init__(self, offsets, number_of_attractive_channels):
        assert HAVE_AFFOGATO, "Need `affogato` module to compute mutex malis loss"
        assert isinstance(offsets, list)
        self.offsets = offsets
        self.number_of_attractive_channels = number_of_attractive_channels

    # can't pickle this by default (probably due to saved tensors ?)
    def __getstate__(self):
        return self.offsets, self.number_of_attractive_channels

    def __setstate__(self, offsets, number_of_attractive_channels):
        self.offsets = offsets
        self.number_of_attractive_channels = number_of_attractive_channels

    def forward(self, input_, target):
        input_ = input_.detach()
        # TODO for multi-gpu training we probably need to remember the cuda devices here
        if input_.is_cuda:
            input_ = input_.cpu()
            is_cuda = True
        else:
            is_cuda = False
        if target.is_cuda:
            target = target.cpu()

        input_ = input_.numpy()
        target = target.numpy()
        assert input_.shape[1] == len(self.offsets), "%i, %i" % (input_.shape[1], self.ndim)
        assert input_.shape[2:] == target.shape[2:], "%s, %s" % (str(input_.shape), str(target.shape))

        normalisation = np.prod(target.shape[1:])

        gradients, loss = [], []
        n_batches = input_.shape[0]
        for batch in range(n_batches):
            ll, gg, _, _ = affl.mutex_malis(input_[batch], target[batch, 0], self.offsets,
                                            self.number_of_attractive_channels)
            # TODO move normalisation somewhere else
            gg /= normalisation
            loss.append(ll)
            gradients.append(gg[None])
        gradients = np.concatenate(gradients, axis=0)

        if is_cuda:
            # TODO for multi-gpu training we probably need to specify the cuda devices
            self.save_for_backward(torch.from_numpy(gradients).cuda())
        else:
            self.save_for_backward(torch.from_numpy(gradients))

        return torch.tensor(sum(loss) / n_batches)

    def backward(self, grad_output):
        gradients, = self.saved_tensors
        return gradients, None
