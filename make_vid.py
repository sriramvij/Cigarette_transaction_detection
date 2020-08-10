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
nframes = 8000
imdir = "./s3_july30/"
vid_fname = "./vids/s3_july30.mp4"
ims = []

for im in os.listdir(imdir):
    ims.append(imdir+im)
    print(":".join(im.split('.')[0].split("_")[-4:]), '%H:%M:%S:%f')
ims.sort(key = lambda date: datetime.strptime(":".join(date.split("/")[-1].split('.')[0].split("_")[-4:]), '%H:%M:%S:%f'))

sz = (400, 300)

fps = 14                    
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                                                   
vout = cv2.VideoWriter()                           
vout.open(vid_fname,fourcc,fps,sz,1)              

print(vid_fname)
   
for i in tqdm(ims[:nframes]):
    frame = cv2.imread(i)
    print(frame.shape)
    if  frame.shape[0] == 300:
        vout.write(frame)
    else:
        pass

vout.release()
