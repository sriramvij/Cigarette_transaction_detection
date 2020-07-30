import numpy as np
import cv2

class detect_transaction:
    """
    This is a class with methods to identify transactions from the video feed.
    first initialize the class, then call the detect method sequentially according to the timestamps. the method will return None when there is no transaction 
    and the input frame where there is a transaction. The ROI is based on the recent set of images where the cigarette packs are in a cardboard box. 
    Change the ROI where needed.
    """
    def __init__(self, history=50, varThreshold=100, detectShadows=True):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        mask = np.uint8(np.greater_equal(fgmask, np.ones_like(fgmask)*128))
        roi = mask[200:300, 20:320,]>0

        if np.sum(roi)>1300:
            mess = 1
        else:
            mess = 0
        
        if mess==1:
            return frame
        else:
            return None