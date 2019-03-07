import numpy as np
import torch
from torch.autograd import Function

try:
    import affogato.learning as affl
    HAVE_AFFOGATO = True
except ImportError:
    HAVE_AFFOGATO = False


class MalisLoss(Function):
    """
    Compute the constrained malis loss.
    """
    def __init__(self, ndim=3):
        assert ndim in (2, 3)
        assert HAVE_AFFOGATO, "Need `affogato` module to compute malis loss"
        self.malis_function = affl.compute_malis_2d if ndim == 2 else affl.compute_malis_3d
        self.ndim = ndim
        if ndim == 2:
            self.offsets = [[-1, 0], [0, -1]]
        else:
            self.offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    # can't pickle this by default (probably due to saved tensors ?)
    def __getstate__(self):
        return self.ndim

    def __setstate__(self, ndim):
        self.ndim = ndim
        if ndim == 2:
            self.offsets = [[-1, 0], [0, -1]]
        else:
            self.offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        self.malis_function = affl.compute_malis_2d if ndim == 2 else affl.compute_malis_3d

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
        assert input_.ndim == self.ndim + 2, "%i, %i" % (input_.ndim, self.ndim + 1)
        assert input_.shape[1] == self.ndim, "%i, %i" % (input_.shape[0], self.ndim)
        assert input_.shape[2:] == target.shape[2:], "%s, %s" % (input_.shape, target.shape)
        gradients, loss = [], []
        n_batches = input_.shape[0]
        for batch in range(n_batches):
            ll, gg = self.malis_function(input_[batch], target[batch, 0], self.offsets)
            loss.append(ll)
            gradients.append(gg[None])
        gradients = np.concatenate(gradients, axis=0)

        if is_cuda:
            # TODO for multi-gpu training we probably need to specify the cuda devices
            self.save_for_backward(torch.from_numpy(gradients).cuda())
        else:
            self.save_for_backward(torch.from_numpy(gradients))

        # TODO in the old malis code, we return the gradients here, I don't get why
        return torch.tensor(sum(loss) / n_batches)

    def backward(self, grad_output):
        gradients, = self.saved_tensors
        return gradients, None
