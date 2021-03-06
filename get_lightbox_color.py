import time

import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)  # constant


# TODO: Implement test color thresholding (Gen?)

class NoBoxError(Exception):

    def __init__(self):
        pass

def crop_image(img_full, gaze_data):
    height, width, channels = img_full.shape
    crop_to_x = .25  # Crop to a fourth of the image
    crop_to_y = .5
    try:
        x_gaze, y_gaze = gaze_data
    except:
        x_gaze = .5
        y_gaze = .5

    x1 = x_gaze - crop_to_x / 2
    x2 = x_gaze + crop_to_x / 2
    if x1 < 0:
        x1 = 0
        x2 = crop_to_x
    elif x2 > 1:
        x1 = 1 - crop_to_x
        x2 = 1

    y1 = y_gaze - crop_to_y / 2
    y2 = y_gaze + crop_to_y / 2
    if y1 < 0:
        y1 = 0
        y2 = crop_to_y
    elif y2 > 1:
        y1 = 1 - crop_to_y
        y2 = 1

    y1 = 1 - y1
    y2 = 1 - y2
    # Crop is [y1:y2, x1:x2]
    img_crop = img_full[int(height * y2):int(height * y1),
                        int(width * x1):int(width * x2)]
    return img_crop


def convert_to_binary_image(img_trans, K):
    Z = img_trans.reshape((-1, 3))
    Z = np.float32(Z)  # convert to np.float32
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

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
    img_binary = cv2.cvtColor(img_bw_rect, cv2.COLOR_BGR2GRAY)

    return img_binary


def find_bounding_box(img_binary, img_crop):
    img_contour, contours, hierarcy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_dim = []
    max_rect = None
    switch_aspect_ratio = float(119)/75  # Aspect ratio of the lightswitch
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        w = min(rect[1])
        h = max(rect[1])
        # Only consider bounding boxes that match our a priori knowledge of light switch dimensions
        if ( (h/w) < (switch_aspect_ratio * 1.27) and ((h/w) > (switch_aspect_ratio * 0.82))):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_crop,[box],0,(0,0,255),2)
            if w*h > max_area:
                max_area = w*h
                max_rect = rect

    cv2.imshow('boxes', img_crop)
    if not max_rect:
        raise NoBoxError

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
    # cv2.imshow('box', img_crop)
    
    # Final cropped & rotated rectangle
    img_lightbox_crop = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
#    cv2.imshow('lightbox', img_lightbox_crop) # Plot what we are going to average the color of

    return img_lightbox_crop
    

def euclidean_distance(gaze_data, img_width, img_height, x, y, w, h):
    x_centroid = x + (w / 2.0)
    y_centroid = y + (h / 2.0)
    gaze_mapped_x = gaze_data[0] * img_width
    gaze_mapped_y = (1 - gaze_data[1]) * img_height
    distance = np.sqrt((x_centroid - gaze_mapped_x)**2 + (y_centroid - gaze_mapped_y)**2)
    return distance


def find_bounding_box_simple(img_binary, img_crop, gaze_data):
    img_contour, contours, hierarcy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width, img_col = img_crop.shape
    # Minimum distance threshold
    min_distance = np.sqrt(img_width ** 2 + img_height ** 2) * .15
    max_dim = []
    # ignore all bounding boxes found touching or very near the edge of the image frame
    img_width_bound_high = img_width * 0.99
    img_width_bound_low = img_width * 0.01
    img_height_bound_high = img_height * 0.99
    img_height_bound_low = img_height * 0.01
    switch_aspect_ratio = float(119) / 75  # Aspect ratio of the light switch
    img_boxes = img_crop.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Only consider bounding boxes that match our a posteriori knowledge of light switch dimensions (w/ parallax)
        if (float(h)/w) < (switch_aspect_ratio * 1.68) and ((float(h)/w) > (switch_aspect_ratio * 0.77)):
            if (h > img_height * 0.065) and (h < img_height * 0.35):
                if (x > img_width_bound_low) and (y > img_height_bound_low) \
                        and (x + w < img_width_bound_high) and (y + h < img_height_bound_high):
                    cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    # print('{0} {1} {2} {3}'.format(x, y, w, h))
                    distance = euclidean_distance(gaze_data, img_width, img_height, x, y, w, h)
                    if distance < min_distance:
                        min_distance = distance
                        max_dim = [x, y, w, h]
    if not max_dim:
        raise NoBoxError

    # x, y, w, h = max_dim
    # cv2.rectangle(img_crop,(x,y),(x+w,y+h),(255, 0, 255),2)
    # cv2.imshow('full image', img_boxes)
    #img_lightbox_crop = img_trans[int(y):int(y+h), int(x):int(x+w)] # Crop down to just the lightswtich
    #cv2.imshow('lightbox', img_lightbox_crop) # Plot what we are going to average the color of

    # Check if the min_distance is reasonably close to box
    if abs(img_width * gaze_data[0] - max_dim[0]) > (max_dim[2] * 2.0):
        raise NoBoxError

    return max_dim


def get_color(dims, img_trans, img_full):
    bw_lightbox = convert_to_binary_image(img_trans[dims[1]:dims[1] + dims[3], dims[0]:dims[0] + dims[2]], 2)
    bw_lightbox = cv2.morphologyEx(bw_lightbox, cv2.MORPH_OPEN, kernel)
    color_slice = img_full[dims[1]:dims[1] + dims[3], dims[0]:dims[0] + dims[2]][bw_lightbox == 0]
    average_color = np.uint8(np.mean(color_slice, axis=0))  # Convert to whole RGB values
    color_swatch = np.zeros((bw_lightbox.shape[0], bw_lightbox.shape[1], 3), np.uint8)

    for height in range(0, bw_lightbox.shape[0]):
        for width in range(0, bw_lightbox.shape[1]):
            if bw_lightbox[height][width] == 0:
                color_swatch[height][width] = average_color

    average_color_swatch = np.array([[average_color] * 100] * 100, np.uint8)  # Make a color swatch
    # cv2.imshow('bw lightbox', bw_lightbox)  # binary version of region of interest (faceplate + switch)
    # cv2.imshow('average color swatch', average_color_swatch)  # color swatch just displaying the average color
    # cv2.imshow('color swatch', color_swatch)  # average color superimposed over the region of interest

    # naive color determination
    color_classification = {0: 'blue', 1: 'green', 2: 'red', 3: 'cream'}  # BGR ordering due to OpenCV
    if abs(int(average_color[1]) - int(average_color[2])) < 10:  # if green and red are within 10 of each other, cream
        main_color = color_classification[3]
    else:
        main_color = color_classification[np.argmax(average_color, axis=0)] # Index of max BGR color determines color
    print("Naive Lightbox guess: {0}, BGR: {1} ".format(main_color, average_color))
    return average_color, main_color


def get_box_color(img_full, gaze_data):
    start_time = time.time()

    # Transform into CIELab colorspace
    img_trans = cv2.cvtColor(img_full, cv2.COLOR_BGR2LAB)
    img_binary = convert_to_binary_image(img_trans, 2)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('binary', img_binary)

    try:
        img_lightbox_dims = find_bounding_box_simple(img_binary, img_full, gaze_data)
    except NoBoxError:
        print('no box found')
        main_color = 'None'
        average_color = [0, 0, 0]  # set to black, since None can cause trouble
        # cv2.waitKey(0)
        return average_color, main_color

    average_color, main_color = get_color(img_lightbox_dims, img_trans, img_full)

    time_taken = time.time() - start_time
    # print(time_taken)

    # cv2.waitKey(0)
    return average_color, main_color

# if __name__ == '__main__':
