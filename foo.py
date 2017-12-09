import cv2
import numpy as np

img = cv2.imread('img.jpg')
height, width, _ = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 150, 200)

cv2.imwrite('img_edges.jpg', edges)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 500)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000000 * (-b))
        y1 = int(y0 + 1000000 * (a))
        x2 = int(x0 - 1000000 * (-b))
        y2 = int(y0 - 1000000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('img_with_line.jpg', img)
