from functools import partial
import numpy as np
import torch
import torch.nn as nn

from inferno.utils.io_utils import yaml2dict
from .simple import SimpleParallelLoader
from .blending import Blending
from .test_time_augmentation import TestTimeAugmenter


class SimpleInferenceEngine(object):

    def __init__(self,
                 model,
                 gpu,
                 blending=None,
                 augmenter=None,
                 crop_padding=False,
                 channel_offsets=None,
                 out_channels=None,
                 num_workers=8):

        assert isinstance(model, nn.Module)
        self.model = model

        if out_channels is None:
            # out channels are defined at different places for
            # u-net / dense u-net
            try:
                self.out_channels = model.out_channels
            except AttributeError:
                self.out_channels = model.output.out_channels
        else:
            self.out_channels = out_channels

        if channel_offsets is not None:
            assert len(channel_offsets) == self.out_channels, "%i, %i" % (len(channel_offsets), self.out_channels)
        self.channel_offsets = channel_offsets

        # TODO validate gpu
        self.gpu = gpu

        if blending is not None:
            assert isinstance(blending, Blending)
        self.blending = blending

        if augmenter is not None:
            assert isinstance(augmenter, TestTimeAugmenter)
        self.augmenter = augmenter

        self.crop_padding = crop_padding
        self.num_workers = num_workers

    @classmethod
    def from_config(cls, config, model):
        config = yaml2dict(config)
        gpu = config.get("gpu", 0)

        blending_config = config.get("blending_config", None)
        blending = None if blending_config is None else \
            Blending(**blending_config)

        # For now we only support loading default TDAs from config
        augmentation_config = config.get("augmentation_config", None)
        augmenter = None if augmentation_config is None else \
            TestTimeAugmenter.default_tda(**augmentation_config)

        crop_padding = config.get("crop_padding", False)
        num_workers = config.get("num_workers", 8)
        offsets = config.get("offsets", None)
        if offsets is not None:
            offsets = [tuple(off) for off in offsets]
        return cls(model,
                   gpu,
                   blending=blending,
                   augmenter=augmenter,
                   crop_padding=crop_padding,
                   num_workers=num_workers,
                   channel_offsets=offsets)

    def get_slicings(self, slicing, shape, padding):
        # crop away the padding (we treat global as local padding) if specified
        # this is generally not necessary if we use blending
        if self.crop_padding:
            # slicing w.r.t the current output
            local_slicing = tuple(slice(pad[0], shape[i] - pad[1])
                                  for i, pad in enumerate(padding))
            # slicing w.r.t the global output
            global_slicing = tuple(slice(slicing[i].start + pad[0],
                                         slicing[i].stop - pad[1])
                                   for i, pad in enumerate(padding))
        # otherwise do not crop
        else:
            local_slicing = np.s_[:]
            global_slicing = slicing
        return local_slicing, global_slicing

    # TODO does not support multi-channel 3d volumes atm
    def apply_model(self, input_, predict_images=False):
        if input_.ndim == 4:
            assert input_.shape[0] == 1
            input_ = input_[0]
        assert input_.ndim == 3, str(input_.ndim)
        with torch.no_grad():
            if predict_images:
                input_tensor = torch.from_numpy(input_[None, :]).cuda(self.gpu)
            else:
                input_tensor = torch.from_numpy(input_[None, None, :]).cuda(self.gpu)
            # This is only necessary in torch < 0.3.
            # we will leave it here for reference until 0.3 support is abandoned in inferno
            # input_tensor = Variable(input_tensor,
            #                         requires_grad=False,
            #                         volatile=True)
            pred = self.model(input_tensor)
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()[0]
            else:
                pred = pred[0].cpu().numpy()[0]
        return pred

    def infer_patch(self, input_, predict_images=False):
        # infer with or without TDA
        if self.augmenter is None:
            # without TDA
            output = self.apply_model(input_, predict_images=predict_images)
        else:
            # with TDA
            output = self.augmenter(input_,
                                    partial(self.apply_model, predict_images=predict_images),
                                    offsets=self.channel_offsets)
        return output

    def infer(self, dataset):
        # build the output volume
        shape = dataset.volume.shape
        output = np.zeros((self.out_channels,) + shape, dtype='float32')
        # loader
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers)
        # mask to count the number of times a pixel was infered
        mask = np.zeros(shape)

        # do the actual inference
        # TODO verbosity and logging
        while True:
            batches = loader.next_batch()
            if not batches:
                print("[*] Inference finished")
                break

            assert len(batches) == 1
            assert len(batches[0]) == 2
            index, input_ = batches[0]
            print("[+] Inferring batch {} of {}.".format(index, len(dataset)))
            print("[*] Input-shape {}".format(input_.shape))

            # get the slicings w.r.t. the current prediction and the output
            local_slicing, global_slicing = self.get_slicings(dataset.base_sequence[index],
                                                              input_.shape,
                                                              dataset.padding)
            output_patch = self.infer_patch(input_)

            if self.blending is not None:
                output_patch, blending_mask = self.blending(output_patch)
                mask[global_slicing] += blending_mask[local_slicing]
            else:
                mask[global_slicing] += 1
            # add slicing for the channel
            global_slicing = (slice(None),) + global_slicing
            local_slicing = (slice(None),) + local_slicing
            # add up predictions in the output
            output[global_slicing] += output_patch[local_slicing]

        # crop padding from the outputs
        # This is only used to crop the dataset in the end
        crop = tuple(slice(pad[0], shape[i] - pad[1]) for i, pad in enumerate(dataset.padding))
        out_crop = (slice(None),) + crop
        output = output[out_crop]

        # divide by the mask to normalize all pixels
        mask = mask[crop]
        assert (mask != 0).all()
        assert mask.shape == output.shape[1:]
        output /= mask

        # return the prediction
        return output

    # for inference with a dataset made up of individual images, not an
    # hdf5 volume
    def infer_images(self, dataset):
        # loader
        loader = SimpleParallelLoader(dataset, num_workers=self.num_workers)
        # list for the output images
        output = [[] for _ in range(len(dataset))]

        # do the actual inference
        # TODO verbosity and logging
        while True:
            batches = loader.next_batch()
            if not batches:
                print("[*] Inference finished")
                break

            assert len(batches) == 1
            assert len(batches[0]) == 2
            index, input_ = batches[0]
            print("[+] Inferring batch {} of {}.".format(index, len(dataset)))
            print("[*] Input-shape {}".format(input_.shape))

            output_patch = self.infer_patch(input_, predict_images=True)
            output[index] = output_patch

        return output
