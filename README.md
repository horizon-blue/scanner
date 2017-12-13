# Scanner
Scans documents using your camera :)

## Installation

```sh
# install python modules
pip3 install -r requirements.txt
# install tesseract (OCR dependency)
# Example: MacOS
# See https://github.com/tesseract-ocr/tesseract for more details
brew install tesseract --with-all-languages
```

## Run the Scanner
```sh
python3 main.py <your_image_file>
```

The output will be stored in `./result/` directory

## Demo
Input Image:
![input](./example/input.jpg)

Boundary Recognition:
![contour](./example/contour.jpg)

Transform using homography:
![transofrm](./example/transofrm.jpg)

Recognize text location:
![text location](./example/overlay.jpg)

Remove text from background:
![background](./example/inpaint.jpg)

Translate and transform the translated text using reverse homography:
![translated](./example/font.jpg)

Merge the text back to the background
![final](./example/final.jpg)