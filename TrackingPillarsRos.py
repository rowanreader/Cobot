#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

# track by colour
# low H, S, V, high H, S, V
# red = [0, 153, 51, 10, 255, 204]
# blue = [70, 204, 51, 109, 255, 204]
# white = [70, 51, 153, 109, 102, 255]
# black = [70, 102, 0, 109, 204, 102]
# yellow = [12, 184, 62, 26, 255, 189]

# in lab
# red = [0, 156, 71, 4, 255, 237]
# blue = [92, 154, 43, 112, 255, 204]
# white = [12, 0, 109, 75, 52, 255]
# black = [0, 0, 0, 178, 177, 80]
# yellow = [18, 108, 85, 34, 255, 243]

red = [0, 152, 0, 5, 255, 255]
blue = [89, 144, 74, 142, 255, 190]
white = [0, 8, 121, 179, 49, 187]
black = [0, 0, 0, 67, 200, 80]
yellow = [10, 141, 131, 34, 255, 255]
# colours to draw bounding box
redCol = (0, 0, 255)
blueCol = (255, 0, 0)
yellowCol = (0, 255, 255)
whiteCol = (255, 255, 255)
blackCol = (0, 0, 0)

rospy.init_node('pillarTracker', anonymous=True)
bridge = CvBridge()

def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)

def dimage_callback(img_msg):
    depth = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    depthImg = np.array(depth, dtype=np.float32)
    return depthImg

def image_callback(img_msg):
    try:
        # need to get depth image here
        # publish to centerpoints topic
        pub = rospy.Publisher('centerpoints', Point, queue_size=10)
        rate = rospy.Rate(2)
        colorImg = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        # height and width of camera frame - check
        frameWidth = 640
        frameHeight = 480
        kernel = np.ones((5, 5), np.uint8)

        hsv = cv2.cvtColor(colorImg, cv2.COLOR_BGR2HSV)
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
        cntsForROS = []
        for c in cnts:
            area = cv2.contourArea(c)
            # print(area)
            if 400 < area < 6000:  # need to change to account for depth
                peri = cv2.arcLength(c, True)
                apprx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(c)
                # within bounding box, find average depth data with outliers removed, if within range of arm (1m?) calculate size
                #temp = depthImg[x:x + h, y:y + w]
                #aveDepth = np.mean(temp)

                # if passes, draw rectangle
                aveDepth = 20 # random
                if 10 < aveDepth < 1500:
                    cv2.putText(colorImg, "(" + str(x) + ", " + str(y) + ", " + str("aveDepth") + ")", (x, y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.putText(img, "Area: " + str(area), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.rectangle(colorImg, (x, y), (x + w, y + h), colours[count], 2)
                    cntsForROS.append([x + w / 2, y + h / 2, aveDepth])
            count += 1

        #cv2.imshow("Frame", colorImg)
        for i in cntsForROS:
            msg = Point()
            msg.x = i[0]
            msg.y = i[1]
            msg.z = i[2]
            pub.publish(msg)
        rate.sleep()
        # k = cv2.waitKey(1)
        # if k == 27:
        #     break

    # cv2.destroyAllWindows()

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    show_image(colorImg)

colorTopic = '/camera/color/image_raw'
# depthTopic = '/camera/depth/color/points'
depthTopic = '/camera/depth/image_rect_raw'
colImg = rospy.Subscriber(colorTopic, Image, image_callback)
dImg = rospy.Subscriber(depthTopic, Image, dimage_callback)
cv2.namedWindow("Image Window", 1)

while not rospy.is_shutdown():
    rospy.spin()





#
# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass