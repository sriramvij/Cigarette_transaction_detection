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

ims = []

for im in os.listdir("./images_july"):
    ims.append("./images_july/"+im)
ims.sort(key = lambda date: datetime.strptime(":".join(date.split("/")[-1].split('.')[0].split("_")[-4:-1]), '%H:%M:%S'))

ims = [[plt.imshow(cv2.imread(i)[:, :, [2, 1, 0]], animated=True)] for i in ims if not isinstance(cv2.imread(i), type(None))]

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                repeat_delay=1000)
ani.save('july_video.mp4')
plt.show()
    
