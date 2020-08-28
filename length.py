import os
from tqdm import tqdm
import numpy as np
import math
import cv2
import time
from datetime import datetime

nframes = -1
imdir = "./store3_aug6/"
ims = []

for im in os.listdir(imdir):
    ims.append(imdir+im)
ims.sort(key = lambda date: datetime.strptime(":".join(date.split("/")[-1].split('.')[0].split("_")[-4:]), '%H:%M:%S:%f'))

print(datetime.strptime(":".join(ims[-1].split("/")[-1].split('.')[0].split("_")[-4:]), '%H:%M:%S:%f')-datetime.strptime(":".join(ims[0].split("/")[-1].split('.')[0].split("_")[-4:]), '%H:%M:%S:%f'),
    len(ims))