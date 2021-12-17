#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import message_filters

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

bridge = CvBridge()

class Sub:
    def __init__(self):

        colorTopic = '/camera/color/image_raw'
        # depthTopic = '/camera/depth/color/points'
        depthTopic = '/camera/aligned_depth_to_color/image_raw'
        self.colImg = message_filters.Subscriber(colorTopic, Image)
        self.dImg = message_filters.Subscriber(depthTopic, Image)
        self.ts = message_filters.TimeSynchronizer([self.colImg, self.dImg], 10)
        self.ts.registerCallback(self.image_callback)
        cv2.namedWindow("Image Window", 1)

    def show_image(self, img):
        cv2.imshow("Image Window", img)
        cv2.waitKey(3)

    # def dimage_callback(img_msg):
    #     depth = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    #     depthImg = np.array(depth, dtype=np.float32)
    #     return depthImg

    def image_callback(self, image1, image2):
        try:
            # need to get depth image here
            # publish to centerpoints topic
            pub = rospy.Publisher('centerpoints', Point, queue_size=10)
            rate = rospy.Rate(6)
            colorImg = bridge.imgmsg_to_cv2(image1, "bgr8")

            image2.encoding = "mono16"
            depth = bridge.imgmsg_to_cv2(image2, desired_encoding='mono8')
            depthImg = np.array(depth, dtype=np.float32)

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

            cnts = cntsRed + cntsBlue + cntsYellow + cntsBlack #+ cntsWhite
            colours = cR + cB + cY + cK #+ cW
            count = 0
            cntsForROS = []

            #rotation for transforming from pixel to world coords
            # -0.3 -0.2 0.1 0 0 0 = x, y, z, yaw, pitch, roll
            # at origin (33 cm away in x direction) width of frame is 36 cm 1 pixel = 0.05625 cm

            translation = [0.014810178428888321, 0.00010297718836227432, 0.00040399961289949715]
            rotation = [[0.9999451637268066, 0.009572459384799004, -0.004250520374625921], [0.009596227668225765, 0.9999382495880127, -0.005607159808278084], [0.004196583293378353, 0.0056476411409676075, 0.9999752640724182]]
            fx = 636.4033203125
            fy = 636.4033203125
            cx = 640.5645751953125
            cy = 356.38800048828125
            scale = 1 # prolly have to fix this one
            fMat = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

            # for color image, remove everything beyond threshold
            gray = 153
            depth3D = np.dstack((depthImg, depthImg, depthImg))
            thresh = 7
            bg_removed = colorImg #np.where((depth3D > thresh) | (depth3D <= 0), gray, colorImg)
            for c in cnts:
                area = cv2.contourArea(c)
                # print(area)
                if 50 < area < 5000:  # need to change to account for depth
                    peri = cv2.arcLength(c, True)
                    apprx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    x, y, w, h = cv2.boundingRect(c)

                    x, y, w, h = 300, 250, 30, 30

                    # within bounding box, find average depth data with outliers removed, if within range of arm (1m?) calculate size
                    #print(int(x+w/2), int(y+ h/2))
                    # print(depthImg.shape)
                    # print(colorImg.shape)
                    temp = depthImg[x:x+w, y:y+h]
                    #print(temp)
                    aveDepth = np.mean(temp) # depth is in mm
                    #print(aveDepth)
                    # rotate x, y, and aveDepth into proper coordinates - aveDepth stays same except for conversion to cm
                    matrix = np.matmul([x*scale, y*scale, scale], np.linalg.inv(fMat))
                    matrix -= translation
                    matrix = np.matmul(matrix, np.linalg.inv(rotation))
                    #print(matrix)
                    worldX = np.round(matrix[0], 1)
                    worldY = np.round(matrix[1], 1)
                    worldDepth = np.round(matrix[2], 1)
                    # also need these
                    worldW = w # np.round(w + translation[0], 1)
                    worldH = h # np.round(h + translation[1], 1)
                    # if passes, draw rectangle
                    #aveDepth = depthImg[x:x+w][y:y+h]/100 # random

                    #if -55 < worldDepth < 350:
                    cv2.putText(bg_removed, "(" + str(worldX) + ", " + str(worldY) + ", " + str(worldDepth) + ")", (x, y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.putText(img, "Area: " + str(area), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.rectangle(bg_removed, (x, y), (x + w, y + h), colours[count], 2)
                    #print(x, y, worldDepth)
                    cntsForROS.append([worldX + (worldW / 2), worldY + (worldH / 2), worldDepth])
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

        # cv2.destroyAllWindows()`

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))



        norm_image = cv2.normalize(depthImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(bg_removed)
        #gray = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        self.show_image(bg_removed)

rospy.init_node('pillarTracker', anonymous=True)
test = Sub()

while not rospy.is_shutdown():
    rospy.spin()



#
# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass
