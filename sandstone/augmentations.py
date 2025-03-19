from abc import ABC, abstractmethod
import random
import numpy as np
import torch

import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from timm.data import create_transform

TRANS_SEP = "@"
ATTR_SEP = "#"


class AbstractAugmentation(ABC):
    """
    Abstract-transformer.
    Default - non cachable
    """

    def __init__(self, args, **kwargs):
        self.args = args
        self._is_cachable = False
        self._caching_keys = ""

    @abstractmethod
    def __call__(self, img, **kwargs):
        pass

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def cachable(self):
        return self._is_cachable

    def set_cachable(self, *keys):
        """
        Sets the transformer as cachable
        and sets the _caching_keys according to the input variables.
        """
        self._is_cachable = True
        name_str = "{}{}".format(TRANS_SEP, self.__class__.__name__)
        keys_str = "".join(ATTR_SEP + str(k) for k in keys)
        self._caching_keys = "{}{}".format(name_str, keys_str)
        return

    def caching_keys(self):
        return self._caching_keys


class TorchvisionToTensor(AbstractAugmentation):
    """Wrapper for torchvision.transforms.ToTensor
    Normalizes the range of the image to [0, 1].
    """

    def __init__(self, args, **extras):
        super().__init__(args)
        self.set_cachable()
        self.transform = T.ToTensor()

    def __call__(self, img, **kwargs):
        return self.transform(img)


class ComposeAug(AbstractAugmentation):
    """
    composes multiple augmentations
    """

    def __init__(self, args, augmentations):
        super(ComposeAug, self).__init__(args)
        self.augmentations = augmentations

    def __call__(self, img, **kwargs):
        for transformer in self.augmentations:
            img = transformer(img, **kwargs)

        return img


class TorchvisionResize(AbstractAugmentation):
    def __init__(self, args, img_size=None, interpolation="cubic", **extras):
        super().__init__(args)
        interpolation_map = {
            "linear": InterpolationMode.BILINEAR,
            "cubic": InterpolationMode.BICUBIC,
        }
        interpolation = interpolation_map[interpolation]
        height, width = img_size
        self.set_cachable(height, width, interpolation)
        self.transform = T.Resize(img_size, interpolation=interpolation)

    def __call__(self, img, **extras):
        return self.transform(img)


class TorchvisionCenterCrop(AbstractAugmentation):
    def __init__(self, args, height=None, width=None, **extras):
        super().__init__(args)
        self.set_cachable(height, width)
        self.transform = T.CenterCrop((height, width))

    def __call__(self, img, **extras):
        return self.transform(img)


class TimmAug(AbstractAugmentation):
    """Timm-based aug used in MAE. Likely no other augmentations are needed with this augmentation."""
    def __init__(self, args, input_size, color_jitter, aa, reprob, remode, recount, mean, std):
        # Note that color_jitter is disabled if autoaugment is on
        self.transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )

    def __call__(self, img):
        return self.transform(img)


class NormalizeTensor2d(AbstractAugmentation):
    """
    torchvision.transforms.Normalize wrapper.
    """

    def __init__(
        self, args, permute=False, channel_means=None, channel_stds=None, **extras
    ):
        super(NormalizeTensor2d, self).__init__(args)
        self.normalize = torchvision.transforms.Normalize(
            torch.Tensor(channel_means), torch.Tensor(channel_stds)
        )
        self.permute = permute

    def transform(self, img):
        if len(img.size()) == 2:
            img = img.unsqueeze(0)
        if self.permute:
            img = img.permute(2, 0, 1)
            return self.normalize(img).permute(1, 2, 0)
        return self.normalize(img)

    def __call__(self, img, **extras):
        if isinstance(img, dict):
            img["input"] = self.transform(img["input"])
            return img
        return self.transform(img)

