import cv2
import numpy as np
from math import floor

# https://github.com/spmallick/learnopencv/blob/master/Homography/utils.py

def mouse_handler(event, x, y, flags, data) :

    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):

    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points

def scale_image_for_display(im_src, WINDOW = 800):
    size = np.array(im_src.shape[0:2])
    scale = WINDOW / np.max(size)
    im_resized = cv2.resize(im_src, (floor(size[1] * scale), floor(size[0] * scale)))
    return im_resized, scale
