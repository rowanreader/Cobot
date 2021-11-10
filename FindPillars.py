import cv2
import numpy as np
from matplotlib import pyplot as plt


def template():
    img_rgb = cv2.imread('singleRedPillar12.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('singleRedPillar1.jpg',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imwrite('res.png',img_rgb)

def similarity():

    img1 = cv2.imread('singleRedPillar1.jpg', 0)
    img2 = cv2.imread('singleRedPillar2.jpg', 0)

    ret, thresh = cv2.threshold(img1, 127, 255, 0)
    ret, thresh2 = cv2.threshold(img2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt1 = contours[0]
    contours, hierarchy = cv2.findContours(thresh2, 2, 1)
    cnt2 = contours[0]

    ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
    print(ret)


template()
# similarity()