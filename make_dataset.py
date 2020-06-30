"""
Make a .npz file as dataset
"""
import os
import numpy as np
from tqdm import tqdm
import json
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


f = open('./Transaction.json', 'r')
f = [json.loads(i) for i in f.readlines()]

data = []
for i in tqdm(f):
    ims = i['content'].split('___')[-1]
    if i['annotation']:
        try:
            label = int(i['annotation']['labels'][0])
            data.append((ims, label))
        except:
            pass
    else:
        pass

data1 = [('./images/'+'_'.join(i[0].split('_')[2:]), i[1]) for i in data if "images_pi" in i[0]]
data2 = [('./images/'+'_'.join(i[0].split('_')[1:]), i[1]) for i in data if "images_pi" not in i[0]]

data = data1 + data2

data = [i for i in data if "beetel" in i[0]]

def resize(image, new_size):
    scale = new_size/np.max(image.shape)
    return cv2.resize(image, (math.floor(image.shape[1]*scale), math.floor(image.shape[0]*scale)))

X = []
Y = []

for i in tqdm(data):

    X.append(resize(cv2.imread(i[0]), 400)[:, :, [2, 1, 0]])
    Y.append(i[1])


Y = np.array(Y)
X = np.array(X)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)
np.savez('train_pi', X=train_X, Y=train_Y)
np.savez('test_pi', X=test_X, Y=test_Y)




