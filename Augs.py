"""
Image augmentation methods and classes
"""

from torchvision import transforms
import numpy as np
import torch
from skimage import io, transform
import math

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        h, w = img.shape[:2]

        if h > w:
            scale = self.output_size/h
            new_h, new_w = scale*h, scale*w
        else:
            scale = self.output_size/w
            new_h, new_w = scale*h, scale*w

        new_h, new_w = math.floor(new_h), math.floor(new_w)

        img = transform.resize(img, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        label = label

        return {'img': img, 'label': label}
    
class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        assert isinstance(p, float) <= 1
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        
        randProb = np.random.random()
        
        if randProb <= self.p:
            img = np.flip(img, 1)
            

        return {'img': img, 'label': label}

class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        assert isinstance(p, float) <= 1
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        
        randProb = np.random.random()
        
        if randProb <= self.p:
            img = np.flip(img, 0)
            

        return {'img': img, 'label': label}

class GaussianNoise(object):

    def __init__(self, p=0.5, mean=0, var=0.1):
        assert isinstance(p, float) <= 1
        self.p = p
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        
        randProb = np.random.random()

        if randProb <= self.p:
            gauss = np.random.normal(self.mean, self.var, img.shape)
            gauss = gauss.reshape(img.shape)
            img = np.clip(img + gauss, 0, 255)

        return {'img': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        img = np.transpose(img, (2, 0, 1))
        return {'img': torch.from_numpy(img),
                'label': label}