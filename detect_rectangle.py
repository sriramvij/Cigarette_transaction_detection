import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sample_img.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = cv2.bitwise_not(gray)

contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for contour in contours[0]:
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

plt.imshow(binary)
plt.show()