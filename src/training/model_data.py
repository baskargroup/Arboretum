#https://github.com/modestyachts/imagenet-testbed/

from typing import Optional, Sequence, Tuple
from io import BytesIO
from PIL import Image
import PIL
import timm

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

model_params = {
'efficientnet-b0': {   'arch': 'efficientnet-b0',
                       'eval_batch_size': 200,
                       'img_size': 224,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b1': {   'arch': 'efficientnet-b1',
                       'eval_batch_size': 200,
                       'img_size': 240,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b2': {   'arch': 'efficientnet-b2',
                       'eval_batch_size': 200,
                       'img_size': 260,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b3': {   'arch': 'efficientnet-b3',
                       'eval_batch_size': 100,
                       'img_size': 300,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b4': {   'arch': 'efficientnet-b4',
                       'eval_batch_size': 100,
                       'img_size': 380,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b5': {   'arch': 'efficientnet-b5',
                       'eval_batch_size': 50,
                       'img_size': 456,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b0-autoaug': {   'arch': 'efficientnet-b0',
                               'eval_batch_size': 200,
                               'img_size': 224,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b1-autoaug': {   'arch': 'efficientnet-b1',
                               'eval_batch_size': 200,
                               'img_size': 240,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b2-autoaug': {   'arch': 'efficientnet-b2',
                               'eval_batch_size': 200,
                               'img_size': 260,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b3-autoaug': {   'arch': 'efficientnet-b3',
                               'eval_batch_size': 100,
                               'img_size': 300,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b4-autoaug': {   'arch': 'efficientnet-b4',
                               'eval_batch_size': 100,
                               'img_size': 380,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b5-autoaug': {   'arch': 'efficientnet-b5',
                               'eval_batch_size': 50,
                               'img_size': 456,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b6-autoaug': {   'arch': 'efficientnet-b6',
                               'eval_batch_size': 25,
                               'img_size': 528,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b7-autoaug': {   'arch': 'efficientnet-b7',
                               'eval_batch_size': 25,
                               'img_size': 600,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b5-randaug': {   'arch': 'efficientnet-b5',
                               'eval_batch_size': 50,
                               'img_size': 456,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b7-randaug': {   'arch': 'efficientnet-b7',
                               'eval_batch_size': 25,
                               'img_size': 600,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b0-advprop-autoaug': {    'arch': 'efficientnet-b0',
                                        'eval_batch_size': 200,
                                        'img_size': 224,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b1-advprop-autoaug': {    'arch': 'efficientnet-b1',
                                        'eval_batch_size': 200,
                                        'img_size': 240,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b2-advprop-autoaug': {    'arch': 'efficientnet-b2',
                                        'eval_batch_size': 200,
                                        'img_size': 260,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b3-advprop-autoaug': {    'arch': 'efficientnet-b3',
                                        'eval_batch_size': 100,
                                        'img_size': 300,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b4-advprop-autoaug': {    'arch': 'efficientnet-b4',
                                        'eval_batch_size': 100,
                                        'img_size': 380,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b5-advprop-autoaug': {    'arch': 'efficientnet-b5',
                                        'eval_batch_size': 50,
                                        'img_size': 456,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b6-advprop-autoaug': {    'arch': 'efficientnet-b6',
                                        'eval_batch_size': 25,
                                        'img_size': 528,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b7-advprop-autoaug': {    'arch': 'efficientnet-b7',
                                        'eval_batch_size': 25,
                                        'img_size': 600,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b8-advprop-autoaug': {    'arch': 'efficientnet-b8',
                                        'eval_batch_size': 25,
                                        'img_size': 672,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]}}

CROP_PADDING = 32
OPENAI_MEAN = [0.485, 0.456, 0.406]
OPENAI_STD = [0.229, 0.224, 0.225]
normalize = Normalize(mean=OPENAI_MEAN, std=OPENAI_STD)

def _convert_to_rgb(image):
    return image.convert('RGB')

def noisystudent_loader():
    model = timm.create_model('tf_efficientnet_l2_ns', pretrained=True)
    transforms = [Resize(800 + CROP_PADDING, interpolation=PIL.Image.BICUBIC), CenterCrop(800)]
    transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    preprocess_train = preprocess_val = Compose(transforms)
    return model, preprocess_train, preprocess_val

def efficientnet_loader(name: str):
    model = timm.create_model(name, pretrained=True)
    params = model_params[name]
    img_size = params['img_size']
    transforms = [Resize(img_size + CROP_PADDING, interpolation=PIL.Image.BICUBIC), CenterCrop(img_size)]
    transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    preprocess_train = preprocess_val = Compose(transforms)
    return model, preprocess_train, preprocess_val