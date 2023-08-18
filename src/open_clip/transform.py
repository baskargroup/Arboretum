from typing import Optional, Sequence, Tuple
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

try:
    from timm.data.auto_augment import rand_augment_transform
except:
    pass
class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')

def _convert_to_grayscale(image):
    image = image.convert('L')
    channels = image.split()
    image_ggg = Image.merge('RGB', (channels[0], channels[0], channels[0]))
    return image_ggg

def _downsample(img):
    temp = BytesIO()
    img.save(temp, format="jpeg", quality=10)
    return Image.open(temp)

def image_transform(
        image_size: int,
        is_train: bool,
        simclr_trans: bool = False,
        downsample_trans: bool = False,
        augreg_trans: bool = False,
        grayscale: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]
    normalize = Normalize(mean=mean, std=std)

    transforms = []
    if simclr_trans and is_train:
        transforms.append(torchvision.transforms.RandomResizedCrop(image_size, 
                                                                   #scale=(0.08, 1.), 
                                                                   scale=(0.7, 1.),
                                                                   interpolation=InterpolationMode.BICUBIC,
                                                                   ))
        transforms.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8))
        transforms.append(torchvision.transforms.RandomGrayscale(p=0.2))
        transforms.append(torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, sigma=(.1, 2.))], p=0.5))
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    if is_train and grayscale:
        transforms.append(_convert_to_grayscale)
    elif is_train:
        transforms.append(_convert_to_rgb)  
    if downsample_trans and is_train:
        transforms.append(_downsample)
    if resize_longest_max:
        transforms.append(ResizeMaxSize(image_size, fill=fill_color))
    else:
        transforms.extend([
            Resize(image_size),
            CenterCrop(image_size),
        ])
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    print("transforms: ", transforms)
    return Compose(transforms)
