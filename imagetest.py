import numpy as np
import cv2
import copy
import time

#TODO: CLEAN!
#TODO: Find color within bounding box
#TODO: Add gaze crop
#TODO: Offset rectange for bounding box
#TODO: Choose bounding box not just by area but nearness to gaze
#TODO: Make function

start_time = time.time()
img_full = cv2.imread('frame0.jpeg')
height, width, channels = img_full.shape 
# Crop is [y1:y2, x1:x2]
img_crop = img_full[int(height*.25):height, int(width*.25):int(width*.75)]

img_trans = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)

Z = img_trans.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
center[0] = [0,0,0]
center[1] = [255,255,255]

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_trans.shape))
res3 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8)
res3 = cv2.morphologyEx(res3, cv2.MORPH_OPEN, kernel) 
res4 = copy.copy(res3)
#cv2.imshow('denoise', res4)

edges = cv2.Canny(res3, 200, 200)

res3, contours, hierarcy = cv2.findContours(res3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

max_area = 0
max_dim = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w*h > max_area:
        max_area = w*h
        max_dim = [x, y, w, h]

x, y, w, h = max_dim
cv2.rectangle(img_crop,(x,y),(x+w,y+h),(0,255,0),2)

time_taken = time.time() - start_time
print(time_taken)
cv2.imshow('box', img_crop)

cv2.waitKey(0)
