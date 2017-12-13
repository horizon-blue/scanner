from homography import document_transformation
from ocr import ocr
from sys import argv, exit
from utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from translate import Translator
import logging
from box_finder import get_rectangle

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

translator = Translator(to_lang='zh')

if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python3 main.py <image_file>")
        exit(1)
    im_src = cv2.imread(argv[1])
    im_resized, scale = scale_image_for_display(im_src)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)
    # pts_src = get_four_points(im_resized)
    # pts_src = pts_src / scale
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

    pts_src = get_rectangle(im_src)

    img = cv2.drawContours(
        im_src.copy(), [pts_src], -1, (255, 180, 50), thickness=10)

    im_resized, _ = scale_image_for_display(img)
    cv2.imshow('Image', im_resized)
    cv2.moveWindow('Image', 30, 0)
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

    pts_src = [(p[0, 0], p[0, 1]) for p in pts_src]

    logger.info('transforming im')
    im_out, im_binary, h = document_transformation(im_src, pts_src)
    cv2.imwrite('result/transofrm.jpg', im_binary)

    # im_resized, _ = scale_image_for_display(im_binary)
    # cv2.imshow('Image', im_resized)
    # cv2.moveWindow('Image', 30, 0)

    logger.info('doing ocr')
    boxes = ocr(im_out)
    # print(boxes[0].position)

    logger.info('creating mask')
    im_box = Image.new(
        'RGB', (im_binary.shape[1], im_binary.shape[0]), 'black')
    draw = ImageDraw.Draw(im_box)
    for box in boxes:
        draw.rectangle(box.position, fill='blue')
    del draw
    im_box = cv2.cvtColor(np.array(im_box), cv2.COLOR_RGB2BGR)
    im_box = cv2.warpPerspective(im_box, np.linalg.inv(
        h), (im_src.shape[1], im_src.shape[0]))

    im_overlay = cv2.addWeighted(im_src, 1, im_box, 0.3, 0)
    cv2.imwrite('result/overlay.jpg', im_overlay)

    logger.info('translating')
    im_font = Image.new(
        'RGB', (im_binary.shape[1], im_binary.shape[0]), 'black')
    draw = ImageDraw.Draw(im_font)
    alpha = 0.8
    for box in boxes:
        print('translating', box.content)
        translation = translator.translate(box.content)

        print(box.position, floor(
            (box.position[1][1] - box.position[0][1]) * alpha))
        font = ImageFont.truetype(
            'NotoSansCJKsc-Medium.otf', floor((box.position[1][1] - box.position[0][1]) * alpha))
        draw.text(box.position[0], translation, font=font, fill='white')
    del draw
    im_font = cv2.cvtColor(np.array(im_font), cv2.COLOR_RGB2BGR)
    im_font = cv2.warpPerspective(im_font, np.linalg.inv(
        h), (im_src.shape[1], im_src.shape[0]))
    im_font = cv2.bitwise_not(im_font)
    cv2.imwrite('result/font.jpg', im_font)

    logger.info('inpainting')
    im_inpaint = cv2.inpaint(im_src, im_box[:, :, 0], 5, cv2.INPAINT_TELEA)
    cv2.imwrite('result/inpaint.jpg', im_inpaint)

    logger.info('blending')
    im_font_gray = cv2.cvtColor(im_font, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(im_font_gray, 127, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(im_inpaint, im_inpaint, mask=mask)
    img_fg = cv2.bitwise_and(im_font, im_font, mask=mask_inv)
    im_mixed = cv2.add(img_bg, img_fg)

    logger.info('finish')
    cv2.imwrite('result/final.jpg', im_mixed)
    # im_resized, _ = scale_image_for_display(im_src2)
    # cv2.imshow('Image', im_src2)

    # cv2.imwrite('out.jpg', im_mixed);

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
