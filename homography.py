import cv2
import numpy as np
from closestpair import closestpair
from math import floor
from utils import *


def document_transformation(im_src, pts_src, w_h = 0.77):
    pts_src_c = np.copy(pts_src)

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

