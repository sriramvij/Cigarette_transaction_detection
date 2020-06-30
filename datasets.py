import cv2
from torch.utils.data import Dataset
from torch import nn
import math



class cigarette(Dataset):

    def __init__(self, dataset, transform=None):

        self.transform = transform
        self.X = dataset['X']
        self.Y = dataset['Y']


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        sample = {'img':self.X[idx], 'label':self.Y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample