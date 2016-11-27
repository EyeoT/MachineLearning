import numpy as np
import cv2
import copy
import time
import os
import csv

#TODO: Add gaze crop
#TODO: Choose bounding box not just by area but nearness to gaze

def read_data(folder_path):
    csv_file_name = 'all_gaze_data.csv'
    csv_file = open(os.path.join(folder_path, csv_file_name), 'r')
    reader = csv.DictReader(csv_file)
    gaze_data = []
    frame_file = ''
    for row in reader:
        if frame_file is not '':
            if row['frame_file'] != frame_file:
                break
        else:
            frame_file = row['frame_file']
        gaze_point = [float(row['x_norm_pos']), float(row['y_norm_pos'])]
        gaze_data.append(gaze_point)
    avg_gaze_pos = np.mean(gaze_data, axis=0)  # Take average of image's gaze position
    return frame_file, gaze_data, avg_gaze_pos


def crop_image(img_full):
    height, width, channels = img_full.shape 
    # Crop is [y1:y2, x1:x2]
    img_crop = img_full[int(height*.25):height, int(width*.25):int(width*.75)]
    return img_crop


def convert_to_binary_image(img_trans):
    Z = img_trans.reshape((-1,3))
    Z = np.float32(Z) # convert to np.float32
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    #Find larger label and color it black
    if np.count_nonzero(label) > len(label)/2:
        center[1] = [0,0,0]
        center[0] = [255,255,255]
    else:
        center[0] = [0,0,0]
        center[1] = [255,255,255]

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    img_bw = center[label.flatten()]
    img_bw_rect = img_bw.reshape((img_trans.shape))
    img_binary = cv2.cvtColor(img_bw_rect,cv2.COLOR_BGR2GRAY)
    return img_binary


def find_bounding_box(img_binary, img_crop):
    img_contour, contours, hierarcy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_dim = []
    switch_aspect_ratio = float(119)/75 # Aspect ratio of the lightswitch
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        w = min(rect[1])
        h = max(rect[1])
        # Only consider bounding boxes that match our a priori knowledge of light switch dimensions
        if ( (h/w) < (switch_aspect_ratio * 1.12) and ((h/w) > (switch_aspect_ratio * 0.82))):
            if w*h > max_area:
                max_area = w*h
                max_rect = rect

    box = cv2.boxPoints(max_rect)
    box = np.int0(box)

    width, height = max_rect[1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = max_rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img_crop, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = min(width, height)
    croppedH = max(width, height)

    cv2.drawContours(img_crop,[box],0,(0,0,255),2)
    cv2.imshow('box', img_crop)
    
    # Final cropped & rotated rectangle
    img_lightbox_crop = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    cv2.imshow('lightbox', img_lightbox_crop) # Plot what we are going to average the color of

    return img_lightbox_crop
    

def find_bounding_box_simple(img_binary, img_crop):
    img_contour, contours, hierarcy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_dim = []
    switch_aspect_ratio = float(119)/75 # Aspect ratio of the lightswitch
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # Only consider bounding boxes that match our a priori knowledge of light switch dimensions
        if ((float(h)/w) < (switch_aspect_ratio * 1.12) and ((float(h)/w) > (switch_aspect_ratio * 0.82))):
            if w*h > max_area:
                max_area = w*h
                max_dim = [x, y, w, h]

    x, y, w, h = max_dim
    cv2.rectangle(img_crop,(x,y),(x+w,y+h),(255, 0, 255),2)
    cv2.imshow('box', img_crop)

    img_lightbox_crop = img_crop[int(y):int(y+h), int(x):int(x+w)] # Crop down to just the lightswtich
    cv2.imshow('lightbox', img_lightbox_crop) # Plot what we are going to average the color of

    return img_lightbox_crop

def get_color(img_lightbox_crop):
    average_row_color = np.mean(img_lightbox_crop, axis=0) # Take average across one dimension of 2D image
    average_color = np.mean(average_row_color, axis=0) # Take average along second dimension, returning final true color avg
    average_color = np.uint8(average_color) # Convert to whole RGB values

    average_color_swatch = np.array([[average_color]*100]*100, np.uint8) # Make a color swatch
    cv2.imshow('color swatch', average_color_swatch) # And display it for debugging

    color_classification = {0 : 'blue', 1 : 'green', 2 : 'red'} # BGR ordering due to OpenCV

    main_color = color_classification[np.argmax(average_color, axis=0)] # Index of max BGR color determines color
    print("Lightbox detected with color {0}!\n".format(main_color))
    return main_color


def get_box_color(frame_file, gaze_data):
    start_time = time.time()
    img_full = cv2.imread(frame_file)

    img_crop = crop_image(img_full)

    # Transform into CIELab colorspace
    img_trans = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)

    img_binary = convert_to_binary_image(img_trans)

    kernel = np.ones((5,5),np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    img_lightbox_crop = find_bounding_box(img_binary, img_crop)

    main_color = get_color(img_lightbox_crop)

    time_taken = time.time() - start_time
    print(time_taken)

    cv2.waitKey(0)
    return main_color


if __name__ == '__main__':
    frame_file, gaze_data, avg_gaze_pos = read_data('')
    get_box_color(frame_file, gaze_data)
