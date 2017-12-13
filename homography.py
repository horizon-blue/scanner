import cv2
import numpy as np
from closestpair import closestpair
from math import floor
from utils import *
from sys import argv, exit


def document_transformation(im_src, pts_src):
    pts_src_c = np.copy(pts_src)

    w_h = 8.5 / 11.0

    width = floor(np.linalg.norm(
        np.diff(np.array(closestpair(pts_src)), axis=0), ord=2))
    height = floor(width / w_h)

    center = np.mean(np.array(pts_src_c), axis=0)
    rads = [(np.arctan2(pt[1] - center[1], pt[0] - center[0]), pt)
            for pt in pts_src_c]
    rads = sorted(rads, key=lambda t: t[0])
    pts = np.array([t[1] for t in rads])

    # im_dst = np.zeros((width, height))
    pts_dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    h, status = cv2.findHomography(pts, pts_dst)

    im_warpped = cv2.warpPerspective(im_src, h, (width, height))

    im_out = im_warpped

    im_lab = cv2.cvtColor(im_warpped, cv2.COLOR_BGR2Lab)

    C = 10
    im_binary = cv2.adaptiveThreshold(im_lab[:, :, 0], 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      floor(width / 3 / 2) * 2 + 1, C)

    return im_out, im_binary, h


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python3 homography.py <image_file>")
        exit(1)
    im_src = cv2.imread(argv[1])
    im_resized, scale = scale_image_for_display(im_src)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)
    pts_src = get_four_points(im_resized)
    cv2.destroyWindow('Image')
    pts_src = pts_src / scale

    im_out, _, _ = document_transformation(im_src, pts_src)

    im_resized, _ = scale_image_for_display(im_out)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
