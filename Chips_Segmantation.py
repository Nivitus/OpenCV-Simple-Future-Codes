import numpy as np
import cv2
import os
from PIL import Image

path = '/home/nivitus/Desktop/chips.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray, 25)
cv2.bilateralFilter(gray,10,50,50)

minDist = 100
param1 = 30
param2 = 50
minRadius = 5
maxRadius = 100
org = (133, 11)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
print(circles[0][0])

if circles is not None:
    circles = np.uint16(np.around(circles))
    # print(circles)
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (247, 242, 240), 2) # x,y and radius values
        rect = cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 128, 255), 2)

    crp = []

    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (247, 242, 240), 2) # x,y and radius values
        rect = cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 128, 255), 2)
        crop = img[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2]]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        low = np.array([0, 63, 66])

        high = np.array([49, 229, 255])

        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(crop, crop, mask=mask)
        
        crp.append(crop)
        cv2.imshow('crop', res)
        cv2.waitKey(0)

    folder = '/home/nivitus/Desktop/cips_folder/'
    for i, crop in enumerate(crp):
        print(i)
        cv2.imwrite(folder+'crop_'+str(i) + '.jpeg', crop)

# cv2.imshow('Chips', img)
# cv2.waitKey(0)
