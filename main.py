from homography import document_transformation
from ocr import ocr
from utils import *

if __name__ == '__main__':
    im_src = cv2.imread('img.jpg')
    im_resized, scale = scale_image_for_display(im_src)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)
    pts_src = get_four_points(im_resized)
    cv2.destroyWindow('Image')
    pts_src = pts_src / scale

    im_out = document_transformation(im_src, pts_src)

    im_resized, _ = scale_image_for_display(im_out)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)

    txt = ocr(im_out)
    print(txt)

    cv2.waitKey(0)

    cv2.destroyAllWindows()