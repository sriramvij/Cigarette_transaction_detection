import os
from tqdm import tqdm
import numpy as np
import cv2
import datetime

timestamp = datetime.datetime.now()
ts = timestamp.strftime("%Y_%m_%d_%I_%M_%S_%f")

# Change Location
fname = f'./vids/{ts}.mp4'

# Change the zero
cap = cv2.VideoCapture(0)


## some videowriter props
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

try:
    os.mkdir("./vids")
except:
    print("vids exists")
    pass

vout = cv2.VideoWriter()
vout.open(fname,fourcc,fps,sz,True)


while True:
    ret, frame = cap.read()

    vout.write(frame)

cap.release()
