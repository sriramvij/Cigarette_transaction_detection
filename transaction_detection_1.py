import numpy as np
import cv2
from tqdm import tqdm


input_vid_1 = './vids/trans_2_cam_1.mp4'
input_vid_2 = './vids/trans_2_cam_2.mp4'
output_vid_1 = "./detected_1.mp4" 
output_vid_2 = "./detected_2.mp4" 


cap1 = cv2.VideoCapture(input_vid_1)
cap2 = cv2.VideoCapture(input_vid_2)

sz1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sz2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = cap1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

vout1 = cv2.VideoWriter()
vout1.open(output_vid_1,fourcc,fps,sz1,1)
vout2 = cv2.VideoWriter()
vout2.open(output_vid_2,fourcc,fps,sz2,1)

frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=100, detectShadows=True)

print("input:", input_vid_1, input_vid_2)
print("output:", output_vid_1, output_vid_2)

for i in tqdm(range(frame_count)):
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    fgmask = fgbg.apply(frame1)

    mask = np.uint8(np.greater_equal(fgmask, np.ones_like(fgmask)*128))

    new_frame1 = frame1*np.stack([mask, mask, mask], axis=-1)
    
    roi = mask[200:300, 20:320,]>0
    
    if np.sum(roi)>1300:
        mess = "1"
    else:
        mess = "0"

    cv2.putText(frame1, mess, org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, color=(0, 255, 0), thickness=2)
    cv2.putText(frame2, mess, org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, color=(0, 255, 0), thickness=2)
    cv2.rectangle(frame1, (20, 200), (320, 300), (255, 0, 0), 2)
    
    vout1.write(frame1)
    vout2.write(frame2)


cap1.release()
cap2.release()

