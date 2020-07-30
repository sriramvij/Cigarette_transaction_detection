import numpy as np
import cv2
from tqdm import tqdm


input_vid = './vids/trans_2.mp4'
output_vid = "./bgr.mp4" 


cap = cv2.VideoCapture(input_vid)

sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))



fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

vout = cv2.VideoWriter()
vout.open(output_vid,fourcc,fps,sz,1)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

prev_frame = np.zeros((sz[1], sz[0], 3))

fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=100, detectShadows=True)

print("input:", input_vid)
print("output:", output_vid)
# while True:
for i in tqdm(range(frame_count)):
    
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    mask = np.uint8(np.greater_equal(fgmask, np.ones_like(fgmask)*128))

    new_frame = frame*np.stack([mask, mask, mask], axis=-1)

    vout.write(new_frame)
    # cv2.imshow("window", frame)
    cv2.waitKey()



cap.release()

