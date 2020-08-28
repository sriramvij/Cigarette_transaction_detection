import os
from tqdm import tqdm
import numpy as np
import math
import cv2
import time
from datetime import datetime

nframes = -1
imdir = "./store3_aug6/"
vid_fname = "./vids/"+imdir.split("/")[1]+".mp4"
ims = []

for im in os.listdir(imdir):
    ims.append(imdir+im)
    # print(":".join(im.split('.')[0].split("_")[-4:]), '%H:%M:%S:%f')
ims.sort(key = lambda date: datetime.strptime(":".join(date.split("/")[-1].split('.')[0].split("_")[-4:]), '%H:%M:%S:%f'))

sz = (500, 375)

fps = 14                    
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                                                   
vout = cv2.VideoWriter()                           
vout.open(vid_fname,fourcc,fps,sz,1)              

print(vid_fname)
   
for i in tqdm(ims[:nframes]):
    frame = cv2.imread(i)
    # print(frame.shape)
    if  frame.shape[0] == 375:
        vout.write(frame)
    else:
        pass

vout.release()
