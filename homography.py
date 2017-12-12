import cv2
import numpy as np
from closestpair import closestpair
from math import floor
from utils import get_four_points

if __name__ == '__main__' :

    im_src = cv2.imread('img.jpg')
    size = np.array(im_src.shape[0:2])
    scale = 500 / np.max(size)
    im_resized = cv2.resize(im_src, (floor(size[1] * scale), floor(size[0] * scale)))
    cv2.imshow("Image", im_resized)
    pts_src = get_four_points(im_resized)
    print(pts_src)
    # pts_src = [[  42,  56 ],
    #            [  46, 468 ],
    #            [ 366, 462 ],
    #            [ 357,  51 ]]
    # pts_src = [(141, 131), (480, 159), (493, 630), (64, 601)]

    pts_src = pts_src / scale
    pts_src_c = np.copy(pts_src)

    w_h = 8.5 / 11.0

    width = floor(np.linalg.norm(np.diff(np.array(closestpair(pts_src)), axis=0), ord=2))
    height = floor(width / w_h)

    center = np.mean(np.array(pts_src_c), axis=0)
    rads = [(np.arctan2(pt[1]-center[1], pt[0]-center[0]), pt) for pt in pts_src_c]
    rads = sorted(rads, key = lambda t: t[0])
    pts = np.array([t[1] for t in rads])

    print(pts)

    im_dst = np.zeros((width, height))
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    h, status = cv2.findHomography(pts, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (width, height))

    # Display images
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)