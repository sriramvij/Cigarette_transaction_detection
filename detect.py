import numpy as np
import cv2
import os
from tqdm import tqdm
from shelf_detection import detect_transaction

input_vid = "./vids/s3_july30.mp4"
output_vid = "./detected.mp4"

cap = cv2.VideoCapture(input_vid)

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

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

vout = cv2.VideoWriter()
vout.open(output_vid,fourcc,fps,sz,True)

detector = detect_transaction(history=50, varThreshold=200, detectShadows=True, threshold=0.001)


print(input_vid)
print(output_vid)
saved_frames = []
prev_frame = None
flag = 0
for i in tqdm(range(frame_count)):

    ret, frame = cap.read()
    if i==0:
        saved_frames.append(frame)
    mess, frame, thres = detector.detect(frame)

    if flag == 1 and mess==0:
        saved_frames.append(frame)
        flag=0
    elif flag==0 and mess==1:
        flag=1

    image = cv2.putText(frame, str(mess), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    
    vout.write(image)

cap.release()

mdp_path = "./mdp_ims/"
try:
    os.mkdir(mdp_path)
except:
    pass
for n,i in tqdm(enumerate(saved_frames)):
    cv2.imwrite(mdp_path+str(n)+".jpg", i)

