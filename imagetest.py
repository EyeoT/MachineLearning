import numpy as np
import cv2
import copy

img_full = cv2.imread('frame0.jpeg')
height, width, channels = img_full.shape 
# Crop is [y1:y2, x1:x2]
img_crop = img_full[int(height*.25):height, int(width*.25):int(width*.75)]
img_bg = copy.copy(img_crop)
img_rg = copy.copy(img_crop)
img_br = copy.copy(img_crop)
img_bg[:,:,2] = 0
img_rg[:,:,0] = 0
img_br[:,:,1] = 0
#cv2.imshow('image',img_crop)

img_trans = cv2.cvtColor(img_bg, cv2.COLOR_BGR2LAB)
cv2.imshow('color blue green', img_trans)
img_trans2 = cv2.cvtColor(img_rg, cv2.COLOR_BGR2LAB)
#cv2.imshow('color red green', img_trans2)
img_trans3 = cv2.cvtColor(img_br, cv2.COLOR_BGR2LAB)
#cv2.imshow('color red blue', img_trans3)


Z = img_trans.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_trans.shape))


cv2.imshow('res2',res2)

cv2.waitKey(0)
