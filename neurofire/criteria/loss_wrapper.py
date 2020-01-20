from numbers import Integral
import numpy as np

import torch.nn as nn

from inferno.utils.python_utils import is_listlike
from .loss_transforms import MaskTransitionToIgnoreLabel
import torch


class BalanceAffinities:
    """
    Compute a weight for different classes, based on the distribution
    """
    def __init__(self, ignore_label=None, offsets=None):
        # if we have an ignore label, we need to instantiate the masking
        # function (which masks all ignore labels and transitions to the ignore label)
        if ignore_label is not None:
            assert isinstance(ignore_label, Integral)
            assert offsets is not None
            self.masker = MaskTransitionToIgnoreLabel(offsets, ignore_label)
        self.ignore_label = ignore_label

    def __call__(self, prediction, target):
        scales = prediction.data.new(*prediction.size()).fill_(1)
        # if we have an ignore label, compute and apply the mask
        if self.ignore_label is not None:
            assert target.size(1) - 1 == prediction.size(1)
            segmentation = target[:, 0:1]
            mask = self.masker.full_mask_tensor(segmentation)
            scales *= mask
            target_affinities = target[:, 1:]
        else:
            assert target.size(1) == prediction.size(1)
            target_affinities = target
        # compute the number of labeled samples and the
        # fraction of positive / negative samples
        n_labeled = scales.sum()
        frac_positive = (scales * target_affinities.data).sum() / n_labeled
        frac_positive = np.clip(frac_positive, 0.05, 0.95)
        frac_negative = 1. - frac_positive
        # compte the corresponding class weights
        # (this is done as in
        # https://github.com/funkey/gunpowder/blob/master/gunpowder/nodes/balance_affinity_labels.py#L47
        # I don't understand exactly why to choose this as weighting)
        w_positive = 1. / (2. * frac_positive)
        w_negative = 1. / (2. * frac_negative)
        weights = prediction.data.new(2)
        weights[0] = w_negative
        weights[1] = w_positive
        return weights


class LossWrapper(nn.Module):
    """
    Wrapper around a torch criterion.
    Enables transforms before applying the criterion.
    Should be subclassed for implementation.
    """
    def __init__(self,
                 criterion,
                 transforms=None,
                 weight_function=None):
        super().__init__()
        # validate: the criterion needs to inherit from nn.Module
        # assert isinstance(criterion, nn.Module)
        self.criterion = criterion
        # validate: transforms need to be callable
        if transforms is not None:
            assert callable(transforms)
        self.transforms = transforms
        if weight_function is not None:
            assert callable(weight_function)
        self.weight_function = weight_function

    def apply_transforms(self, prediction, target):
        # check if the tensors (prediction and target are lists)
        # if so , we need to loop and apply the transforms to each element inidvidually
        is_listlike = isinstance(prediction, (list, tuple))
        if is_listlike:
            assert isinstance(target, (list, tuple))
        # list-like input
        if is_listlike:
            transformed_prediction, transformed_target = [], []
            for pred, targ in zip(prediction, target):
                tr_pred, tr_targ = self.transforms(pred, targ)
                transformed_prediction.append(tr_pred)
                transformed_target.append(tr_targ)
        # tensor input
        else:
            transformed_prediction, transformed_target = self.transforms(prediction, target)
        return transformed_prediction, transformed_target

    def forward(self, prediction, target):
        # calculate the weight based on prediction and target
        if self.weight_function is not None:
            weight = self.weight_function(prediction, target)
            self.criterion.weight = weight

        # apply the transforms to prediction and target or a list of predictions and targets
        if self.transforms is not None:
            prediction, target = self.apply_transforms(prediction, target)

        loss = self.criterion(prediction, target)
        return loss


class MultiOutputLossWrapper(nn.Module):
    """
    Wrapper around a torch criterion.
    Enables transforms before applying the criterion.
    Expects a list of tensors as input and returns the sum of loss over all elements.
    """
    def __init__(self,
                 criterion,
                 transforms=None):
        super(MultiOutputLossWrapper, self).__init__()
        # validate: the criterion needs to inherit from nn.Module
        assert isinstance(criterion, nn.Module)
        self.criterion = criterion
        # validate: transforms need to be callable
        if transforms is not None:
            assert callable(transforms)
        self.transforms = transforms
        self.weight = 1

    def slice_loss(self, pred, target):
        if self.transforms is not None:
            transformed_prediction, transformed_target = self.transforms(pred, target)
        else:
            transformed_prediction, transformed_target = pred, target
        return self.criterion(transformed_prediction, transformed_target)

    def forward(self, predictions, target):
        loss = 0
        assert isinstance(predictions, (list, tuple))
        for pred in predictions[:-1]:
            if self.weight > 0:
                loss = loss + self.weight * self.slice_loss(pred, target)

        loss = loss + self.slice_loss(predictions[-1], target)
        return loss
