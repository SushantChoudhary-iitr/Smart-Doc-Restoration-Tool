import cv2
import numpy as np

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    #img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0,-1,0],
              [-1,5,-1],
              [0,-1,0]])

    sharpen = cv2.filter2D(gray, -1, kernel)

    blur = cv2.GaussianBlur(sharpen, (5,5), 0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    h, w = thresh.shape
    cropped = thresh[int(h*0.1):h, 0:w]

    return cropped