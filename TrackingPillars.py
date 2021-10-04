import cv2
import numpy as np
import pyrealsense2 as rs

def empty(img):
    pass

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# min framerate is 6
frameRate = 60
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frameRate)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, frameRate)
pipeline.start(config)

# height and width of camera frame - check
frameWidth = 640;
frameHeight = 480;

maxObj = 50; # can only detect up to 50 objects
minObjArea = 10*10 # minimum object area
maxObjArea = frameHeight*frameWidth/1.5 # if larger than this, can't detect

# track by colour
# low H, S, V, high H, S, V
# red = [0, 153, 51, 10, 255, 204]
# blue = [70, 204, 51, 109, 255, 204]
# white = [70, 51, 153, 109, 102, 255]
# black = [70, 102, 0, 109, 204, 102]
# yellow = [12, 184, 62, 26, 255, 189]

# in lab
red = [0, 156, 71, 4, 255, 237]
blue = [92, 154, 43, 112, 255, 204]
white = [12, 0, 109, 75, 52, 255]
black = [0, 0, 0, 178, 177, 80]
yellow = [18, 108, 85, 34, 255, 243]


# colours to draw bounding box
redCol = (0, 0, 255)
blueCol = (255, 0, 0)
yellowCol = (0, 255, 255)
whiteCol = (255, 255, 255)
blackCol = (0, 0, 0)

# video = cv2.VideoCapture(2)

while True:
    # depth part
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_HOT)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = img.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(img, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((img, depth_colormap))



    # colour camera part
    # ret, img = video.read()

    kernel = np.ones((5, 5), np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerRed = np.array([red[0], red[1], red[2]])
    upperRed = np.array([red[3], red[4], red[5]])
    maskRed = cv2.inRange(hsv, lowerRed, upperRed)
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernel)
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel)


    cntsRed, heiR = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cR = [redCol for _ in cntsRed]

    lowerBlue = np.array([blue[0], blue[1], blue[2]])
    upperBlue = np.array([blue[3], blue[4], blue[5]])
    maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel)
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel)

    cntsBlue, heiB = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cB = [blueCol for _ in cntsBlue]

    lowerYellow = np.array([yellow[0], yellow[1], yellow[2]])
    upperYellow = np.array([yellow[3], yellow[4], yellow[5]])
    maskYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel)
    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel)

    cntsYellow, heiY = cv2.findContours(maskYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cY = [yellowCol for _ in cntsYellow]

    lowerBlack = np.array([black[0], black[1], black[2]])
    upperBlack = np.array([black[3], black[4], black[5]])
    maskBlack = cv2.inRange(hsv, lowerBlack, upperBlack)
    maskBlack = cv2.morphologyEx(maskBlack, cv2.MORPH_OPEN, kernel)
    maskBlack = cv2.morphologyEx(maskBlack, cv2.MORPH_CLOSE, kernel)

    cntsBlack, hei2 = cv2.findContours(maskBlack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cK = [blackCol for _ in cntsBlack]

    lowerWhite = np.array([white[0], white[1], white[2]])
    upperWhite = np.array([white[3], white[4], white[5]])
    maskWhite = cv2.inRange(hsv, lowerWhite, upperWhite)
    maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_OPEN, kernel)
    maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_CLOSE, kernel)

    cntsWhite, hei2 = cv2.findContours(maskWhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cW = [whiteCol for _ in cntsWhite]


    cnts = cntsRed + cntsBlue + cntsYellow + cntsBlack + cntsWhite
    colours = cR + cB + cY + cK + cW
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        # print(area)
        if 400 < area < 6000: # need to change to account for depth
            peri = cv2.arcLength(c, True)
            apprx = cv2.approxPolyDP(c, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(c)
            # within bounding box, find average depth data with outliers removed, if within range of arm (1m?) calculate size
            temp = depth_image[x:x+h, y:y+w]
            # remove outliers to calc depth - poorer results
            # std = np.std(temp)
            # lowerBound = np.mean(temp) - 3*std
            # upperBound = np.mean(temp) + 3*std
            # temp = temp.flatten()
            # temp2 = [a for a in temp if (lowerBound < a and a < upperBound)]
            # aveDepth = np.mean(temp2)

            aveDepth = np.mean(temp)

            # if passes, draw rectangle
            if 10 < aveDepth < 1500 :
                cv2.putText(img, "(" + str(x) + ", " + str(y) + ", " + str(aveDepth) + ")", (x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
            # cv2.putText(img, "Area: " + str(area), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                cv2.rectangle(img, (x, y), (x + w, y + h), colours[count], 2)
            # cv2.putText(img, "Points: " + str(len(apprx)), (x+w+20, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0),2)
            # if len(apprx) == 4:
            #     x, y, w, h = cv2.boundingRect(c)
            #     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 2)
        count += 1

    cv2.imshow("Frame", img)

    # cv2.imshow("Depth", depth_image)
    # cv2.imshow("White Mask", maskWhite)
    # cv2.imshow("Black Mask", maskBlack)
    k = cv2.waitKey(1)
    if k == 27:
        break

pipeline.stop()
# video.release()
cv2.destroyAllWindows()