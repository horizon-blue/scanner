import cv2
import numpy as np
from utils import *
from sys import argv, exit

# with some reference to
# https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection


def get_rectangle(im_src):
    # smooth the image with medianBlur filter
    img = cv2.medianBlur(im_src, 11)
    # find the edge in each of the channels, keep only the
    # maximum among three channels
    img = cv2.max(*[cv2.Canny(channel, 10, 50)
                    for channel in cv2.split(img)])

    # close the tiny openings of line
    # use horizontal kernel
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((1, 20)))
    # follow by a vertical kernel
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((20, 1)))

    # find the out-most contour
    _, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for debug:
    # img = cv2.drawContours(im_src, contours, 0, (255, 180, 50), thickness=10)

    # find rectangles
    maxarea = -np.inf
    largest_rect = None
    for contour in contours:
        curve = cv2.approxPolyDP(
            contour, cv2.arcLength(contour, True) * 0.02, True)
        # only save 4 edges polygon of maximum area
        curve_area = cv2.contourArea(curve)
        if len(curve) == 4 and curve_area > maxarea:
            maxarea = curve_area
            largest_rect = curve

    return largest_rect


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python3 box_finder.py <image_file>")
        exit(1)
    im_src = cv2.imread(argv[1])
    im_resized, scale = scale_image_for_display(im_src)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

    largest_rect = get_rectangle(im_src)

    img = cv2.drawContours(
        im_src, [largest_rect], -1, (255, 180, 50), thickness=10)

    im_resized, _ = scale_image_for_display(img)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
