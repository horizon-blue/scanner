from PIL import Image
import sys

import pyocr
import pyocr.builders

def ocr(im):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]
    # print("Will use tool '%s'" % (tool.get_name()))

    # langs = tool.get_available_languages()
    lang = 'eng'
    # print("Will use lang '%s'" % (lang))

    return tool.image_to_string(
        Image.fromarray(im),
        lang=lang,
        builder=pyocr.builders.LineBoxBuilder()
    )