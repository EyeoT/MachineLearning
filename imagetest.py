import numpy as np
import cv2
import copy
import time

#TODO: CLEAN!
#TODO: Add gaze crop
#TODO: Offset rectange for bounding box
#TODO: Choose bounding box not just by area but nearness to gaze

def get_box_color():
    start_time = time.time()
    img_full = cv2.imread('frame0.jpeg')
    height, width, channels = img_full.shape 
    # Crop is [y1:y2, x1:x2]
    img_crop = img_full[int(height*.25):height, int(width*.25):int(width*.75)]

    # Transform into CIELab colorspace
    img_trans = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)

    Z = img_trans.reshape((-1,3))
    Z = np.float32(Z) # convert to np.float32
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    # Convert colors to binary
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
    cv2.imshow('denoise', res4)

    edges = cv2.Canny(res3, 200, 200)

    res3, contours, hierarcy = cv2.findContours(res3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    max_area = 0
    max_dim = []
    switch_aspect_ratio = (119/75) # Aspect ratio of the lightswitch
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # Only consider bounding boxes that match our a priori knowledge of light switch dimensions
        if ( (h/w) < (switch_aspect_ratio * 1.02) and ((h/w) > (switch_aspect_ratio * 0.92))):
            if w*h > max_area:
                max_area = w*h
                max_dim = [x, y, w, h]

    x, y, w, h = max_dim

    img_lightbox_crop = img_crop[int(y):int(y+h), int(x):int(x+w)] # Crop down to just the lightswtich
    cv2.imshow('lightbox', img_lightbox_crop) # Plot what we are going to average the color of

    cv2.rectangle(img_crop,(x,y),(x+w,y+h),(255, 0, 255),2)
    cv2.imshow('box', img_crop)

    average_row_color = np.mean(img_lightbox_crop, axis=0) # Take average across one dimension of 2D image
    average_color = np.mean(average_row_color, axis=0) # Take average along second dimension, returning final true color avg
    average_color = np.uint8(average_color) # Convert to whole RGB values

    average_color_swatch = np.array([[average_color]*100]*100, np.uint8) # Make a color swatch
    cv2.imshow('color swatch', average_color_swatch) # And display it for debugging

    color_classification = {0 : 'blue', 1 : 'green', 2 : 'red'} # BGR ordering due to OpenCV

    main_color = color_classification[np.argmax(average_color, axis=0)] # Index of max BGR color determines color
    print("Lightbox detected with color {0}!\n".format(main_color))

    time_taken = time.time() - start_time
    print(time_taken)

    cv2.waitKey(0)
    return main_color


if __name__ == '__main__':
    get_box_color()
