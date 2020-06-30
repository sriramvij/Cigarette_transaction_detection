"""
Make training dataset
"""
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import numpy as np
import math
import cv2
import scipy
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch import nn
from Augs import *
from datasets import cigarette


train = np.load('train_pi.npz')
test = np.load('test_pi.npz')

print('='*50)
print('Make Datasets')
print('='*50)
train_dataset = cigarette(dataset=train, transform=transforms.Compose([
                                               RandomHorizontalFlip(p=0.5),
                                               RandomVerticalFlip(p=0.5),
                                               Rescale(400),
                                               GaussianNoise(p=0.5, mean=0, var=0.1),
                                               ToTensor()]))
test_dataset = cigarette(dataset=test, transform=transforms.Compose([
                                               Rescale(400),
                                               ToTensor()]))


print('='*50)
print('Make DataLoaders')
print('='*50)
device = "cuda" if  torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 16, 'pin_memory': True} if device=='cuda' else {}

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)

print('='*50)
print('Init Models')
print('='*50)

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = nn.Sequential(model, nn.Sigmoid())
model = model.to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.1, patience=10, verbose=True, 
                                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

EPOCHS = 250

print('='*50)
print('Model Training')
print('='*50)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    print("Epoch:", epoch)
    for i in tqdm(train_loader):
        img, label = i['img'], i['label']
        img = img.to(device).float()
        label = label.to(device).float().unsqueeze(-1)

        optimizer.zero_grad()

        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(running_loss)
    if epoch == 0:
        prev_loss = running_loss
    else:
        if running_loss < prev_loss:
            print(str(running_loss)+' < '+str(prev_loss))
            fname = 'model_best.pth'
            print('Saving Model')
            torch.save(model, fname)
            prev_loss = running_loss
    lr_scheduler.step(running_loss)

print('='*50)
print('Save Final Model')
print('='*50)    

torch.save(model, "./model.pth")

print('='*50)
print('End')
print('='*50)
