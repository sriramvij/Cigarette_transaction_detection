import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import cv2
import time
from datetime import datetime
import matplotlib.animation as animation

fig = plt.figure()
nframes = 5000
imdir = "./images_temp/"
vid_fname = "./vids/trans_2_cam_2.mp4"
ims = []

for im in os.listdir(imdir):
    ims.append(imdir+im)
ims.sort(key = lambda date: datetime.strptime(date.split("/")[-1].split('.')[0].split("_")[-1], '%H:%M:%S:%f'))

sz = (400, 225)

fps = 14                    
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                                                   
vout = cv2.VideoWriter()                           
vout.open(vid_fname,fourcc,fps,sz,1)              

print(vid_fname)
   
for i in tqdm(ims[10000:]):
    frame = cv2.imread(i)
    if frame.shape[0] == 225:
        vout.write(frame)

vout.release()
