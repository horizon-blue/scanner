from PIL import Image
import sys

import pyocr
import pyocr.builders

def ocr(im):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = 'eng'
    print("Will use lang '%s'" % (lang))

    return tool.image_to_string(
        Image.fromarray(im),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )