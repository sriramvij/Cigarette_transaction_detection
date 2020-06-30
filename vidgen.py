"""
Make prediction on video
"""
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import cv2
import scipy
from torchvision import transforms
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
from torch import nn
import shutil
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime

#video parameters
file_name = './test_vids/test_vid6.mp4'
cap = cv2.VideoCapture(file_name)

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 2

## some videowriter props
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# run model on cpu instead of gpu
device = "cpu"

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

vout = cv2.VideoWriter()
vout.open('./output_new.mp4',fourcc,fps,sz,True)

# load model
model = torch.load('./model_best.pth')
model.to(device)

# Disable gradient tape
model.eval()

trans_count = 0
pred_mem = 0
hold = False
hold_dur = 50   # Holdout switch duration
hold_count = 0

# Background removing filter
fgbg = cv2.createBackgroundSubtractorMOG2()

for i in tqdm(range(frame_count)):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    mask = np.uint8(np.greater_equal(fgmask, np.ones_like(fgmask)*128))

    new_frame = frame[:, :, [2, 1, 0]]*np.stack([mask, mask, mask], axis=-1)

    img = torch.Tensor([cv2.resize(new_frame, (400, 225))]).permute(0, -1, 1, 2)/255
    img = img.to(device)

    # Hold is the timeout switch. This makes it check only every 50 frames
    if not hold:
        pred = float(model(img).detach().to('cpu')[0][0])>0.95
        if pred != pred_mem:
            if pred==0 and pred_mem==1:
                trans_count += 1
            pred_mem = pred
        hold = not hold
    
    message = "{:.2f}".format(pred*100)+"%  "+str(trans_count)


    image = cv2.putText(frame, message, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    
    # Write frames to video file 
    vout.write(image)

    # Toggle swtich after holdout period
    hold_count += 1
    if hold_count >= hold_dur:
        hold_count = 0
        hold = not hold

cap.release()
