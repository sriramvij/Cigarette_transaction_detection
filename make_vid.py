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

for im in os.listdir("./04_13"):
    ims.append("./04_13/"+im)
ims.sort(key = lambda date: datetime.strptime(date.split("/")[-1].split('.')[0].split("_")[-1], '%H:%M:%S:%f'))

# print(len(ims))

# for i in ims:
#     print(i)

ims = [[plt.imshow(cv2.imread(i)[:, :, [2, 1, 0]], animated=True)] for i in ims[:] if not isinstance(cv2.imread(i), type(None))]

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('04_13.mp4')
plt.show()
    
