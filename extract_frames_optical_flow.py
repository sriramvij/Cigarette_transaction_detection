"""
Extract frames from training videos to make training dataset.
"""


import numpy as np
import cv2
import os
from tqdm import tqdm

for vid in os.listdir('./vids/'):
    vidcap = cv2.VideoCapture('./vids/'+vid)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fname = vid.split(".")[0]
    count = 0
    print("Extracting from:", vid)
    print(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    try:
        for i in tqdm(range(frame_count)):
            success, frame = vidcap.read()

            fgmask = fgbg.apply(frame)

            mask = np.uint8(np.greater_equal(fgmask, np.ones_like(fgmask)*128))

            new_frame = frame*np.stack([mask, mask, mask], axis=-1)

            cv2.imwrite("./images/"+fname+"_frame_%d.jpg" % count, new_frame)     # save frame as JPEG file
            count += 1
    except:
        print("error")
        pass
