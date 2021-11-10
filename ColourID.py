import pyrealsense2 as rs

# """
# File: opencv-webcam-object-detection.py
#
# This Python 3 code is published in relation to the article below:
# https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
#
# Website:    www.bluetin.io
# Author:     Mark Heywood
# Date:	    31/12/2017
# Version     0.1.0
# License:    MIT
# """
#
# from __future__ import division
# import cv2
# import numpy as np
# import time
#
#
# def nothing(*arg):
#     pass
#
#
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 240
#
# # Initial HSV GUI slider values to load on program start.
# # icol = (36, 202, 59, 71, 255, 255)    # Green
# # icol = (18, 0, 196, 36, 255, 255)  # Yellow
# # icol = (89, 0, 0, 125, 255, 255)  # Blue
# # icol = (0, 100, 80, 10, 255, 255)   # Red
# # icol = (104, 117, 222, 121, 255, 255)   # test
# icol = (0, 0, 0, 179, 255, 255)  # New start
#
# cv2.namedWindow('colorTest')
# # Lower range colour sliders.
# cv2.createTrackbar('lowHue', 'colorTest', icol[0], 179, nothing)
# cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
# cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# # Higher range colour sliders.
# cv2.createTrackbar('highHue', 'colorTest', icol[3], 179, nothing)
# cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
# cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)
#
# # Initialize webcam. Webcam 0 or webcam 1 or ...
# vidCapture = cv2.VideoCapture(0)
# vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#
# while True:
#     timeCheck = time.time()
#     # Get HSV values from the GUI sliders.
#     lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
#     lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
#     lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
#     highHue = cv2.getTrackbarPos('highHue', 'colorTest')
#     highSat = cv2.getTrackbarPos('highSat', 'colorTest')
#     highVal = cv2.getTrackbarPos('highVal', 'colorTest')
#
#     # Get webcam frame
#     _, frame = vidCapture.read()
#
#     # Show the original image.
#     cv2.imshow('frame', frame)
#
#     # Convert the frame to HSV colour model.
#     frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # HSV values to define a colour range we want to create a mask from.
#     colorLow = np.array([lowHue, lowSat, lowVal])
#     colorHigh = np.array([highHue, highSat, highVal])
#     mask = cv2.inRange(frameHSV, colorLow, colorHigh)
#     # Show the first mask
#     cv2.imshow('mask-plain', mask)
#
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
#     biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
#
#     # cv2.drawContours(frame, biggest_contour, -1, (0,255,0), 3)
#
#     x, y, w, h = cv2.boundingRect(biggest_contour)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # cv2.drawContours(frame, contours, -1, (0,255,0), 3)
#
#     # cv2.drawContours(frame, contours, 3, (0,255,0), 3)
#
#     # cnt = contours[1]
#     # cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
#
#     # Show final output image
#     cv2.imshow('colorTest', frame)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#     print('fps - ', 1 / (time.time() - timeCheck))
#
# cv2.destroyAllWindows()
# vidCapture.release()


import cv2
import numpy as np


# 0 is laptops, 2 is usb
# cap = cv2.VideoCapture(2)
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)

cv2.createTrackbar('h2', 'result',179,179,nothing)
cv2.createTrackbar('s2', 'result',255,255,nothing)
cv2.createTrackbar('v2', 'result',255,255,nothing)

#
setNeutral = [0,0,0,179,255,255]
setRed = [0, 105, 129, 9, 255, 255]
setWhite = [12, 0, 109, 75, 52, 255]
setBlack = [0,0,0, 104, 74, 98]
setYellow = [18, 108, 85, 34, 255, 243]

setCol = setRed
cv2.setTrackbarPos('h', 'result', setCol[0])
cv2.setTrackbarPos('s', 'result', setCol[1])
cv2.setTrackbarPos('v', 'result', setCol[2])
cv2.setTrackbarPos('h2', 'result', setCol[3])
cv2.setTrackbarPos('s2', 'result', setCol[4])
cv2.setTrackbarPos('v2', 'result', setCol[5])


while(1):

    # test, frame = cap.read()
    # if test == True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # cv2.imshow('test', frame)
    #converting to HSV
    hsv = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    lowH = cv2.getTrackbarPos('h','result')
    lowS = cv2.getTrackbarPos('s','result')
    lowV = cv2.getTrackbarPos('v','result')

    upH = cv2.getTrackbarPos('h2', 'result')
    upS = cv2.getTrackbarPos('s2', 'result')
    upV = cv2.getTrackbarPos('v2', 'result')

    # Normal masking algorithm
    lower_blue = np.array([lowH,lowS,lowV])
    upper_blue = np.array([upH,upS,upV])


    mask = cv2.inRange(hsv,lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, hei2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        area = cv2.contourArea(c)
        # print(area)
    result = cv2.bitwise_and(color_image, color_image, mask = mask)
    result = cv2.resize(result, (760, 340))
    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # else:
    #     print("Camera not found")
    #     break

# cap.release()
#
# cv2.destroyAllWindows()

pipeline.stop()