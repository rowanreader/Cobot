#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
import rospy
import pcl
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
import message_filters
import ros_numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roslib import message
from geometry_msgs.msg import Transform, Vector3, Quaternion
import time

# gets the o, x, y, and z from mouse clicking
bridge = CvBridge()
# points defining origin, x axis, y axis, and z axis
o = 0
x = 0
y = 0
z = 0
counter = 0
flag = 0
landmarks = []

def bubble(areas, contours):
    n = len(areas)
    for i in range(n):
        for j in range(n-i-1):
            if areas[j] > areas[j+1]:
                areas[j], areas[j+1] = areas[j+1], areas[j]
                contours[j], contours[j+1] = contours[j+1], contours[j]
    return contours

class Sub:
    def __init__(self):

        colorTopic = '/camera/color/image_raw'
        depthTopic = '/camera/aligned_depth_to_color/image_raw'
        # ptsTopic = '/camera/depth/color/points'
        ptsTopic = '/camera/depth_registered/points'
        self.colImg = message_filters.Subscriber(colorTopic, Image)
        self.pts = message_filters.Subscriber(ptsTopic, PointCloud2)
        self.dImg = message_filters.Subscriber(depthTopic, Image)
        self.ts = message_filters.TimeSynchronizer([self.colImg, self.dImg, self.pts], 10)
        self.ts.registerCallback(self.image_callback)
        cv2.namedWindow("Image Window", 1)

    def generateTransform(self, landmarks):
        # don't need origin
        dx = landmarks[1]
        dy = landmarks[2]
        dz = landmarks[3]

        R = [dx, dy, dz]
        t = landmarks[0]

        xy = np.matmul(dx, dy)
        yz = np.matmul(dy, dz)
        zx = np.matmul(dz, dx)

        thxy = np.degrees(np.arccos(xy))
        thyz = np.degrees(np.arccos(yz))
        thzx = np.degrees(np.arccos(zx))

        return np.transpose(R), t, [thxy, thyz, thzx]

    def robotToCamera(self, pr, R, t):
        pc = np.matmul(R, pr) + t
        # print("R: " + str(R))
        return pc

    def cameraToRobot(self, pc, R, t):
        pr = np.matmul(np.linalg.inv(R), pc - t)
        return pr


    def show_image(self, img):
        cv2.imshow("Image Window", img)
        cv2.setMouseCallback('Image Window', self.click_event)
        cv2.waitKey(3)

    def click_event(self, event, x, y, flags, params):
        global mousex, mousey, flag, counter, colorImg
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(colorImg, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            mousex = x
            mousey = y
            counter += 1
            flag = 1 # if flag is 1, record and rest flag to 0
            print(x, y)

    def getPoint(self, points, x, y):
        # x goes up to 640, y goes up to 480
        # do num times, get median
        num = 100
        pList = np.zeros((num, 3))
        for i in range(num):
            pc1 = pc2.read_points(points, skip_nans=False, field_names=("x", "y", "z"), uvs=[(x, y)])
            p = list(pc1)[0]
            pList[i] = p
        pList = np.sort(pList, axis = -1)

        return pList[num//2]

    # given a 2D array of points, plot
    def plotPts(self, ax, points):
        px = []
        py = []
        pz = []
        for i in points:
            px.append(i[0])
            py.append(i[1])
            pz.append(i[2])
            # break # only plots 1st point
        # ax.scatter(px, py, pz, color=['c', 'm', 'y'], alpha=1)
        ax.scatter(px, py, pz, alpha=1)


    # given 4 points in 3D, plot coord
    # 1st is origin, then x, y, z
    def plotCoords(self, ax, points):
        px = []
        py = []
        pz = []
        for i in points:
            px.append(i[0])
            py.append(i[1])
            pz.append(i[2])
        colours = ['r', 'b', 'g']
        for i in range(3):
            ax.plot([px[0], px[i+1]], [py[0], py[i+1]], [pz[0], pz[i+1]], colours[i])


    # given rotation matrix, convert to quaternion form
    def convertToQuat(self, R):
        w = np.sqrt(1 + R[0][0] + R[1][1] + R[2][2])/2
        x = (R[2][1] - R[1][2])/(4*w)
        y = (R[0][2] - R[2][0])/(4*w)
        z = (R[1][0] - R[0][1])/(4*w)
        print("w: " + str(w))
        return [x, y, z, w]




    def image_callback(self, image1, image2, points):

        try:
            pub = rospy.Publisher('transformation', Transform, queue_size=1)
            rate = rospy.Rate(6)


            # colour profiles fro corners
            pink = [151, 25, 93, 178, 161, 245]
            blue = [73, 128, 66, 179, 255, 194]
            # yellow = [13, 119, 75, 57, 255, 255]
            # yellow is now red
            yellow = [0, 99, 189, 179, 234, 255]
            green = [29, 59, 77, 62, 107, 223]
            colorImg = bridge.imgmsg_to_cv2(image1, "bgr8")
            # convert to HSV
            hsvImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2HSV)
            # using the colour profiles, get contours of colours

            kernel = np.ones((5, 5), np.uint8)

            lowerPink = np.array([pink[0], pink[1], pink[2]])
            upperPink = np.array([pink[3], pink[4], pink[5]])
            maskPink = cv2.inRange(hsvImg, lowerPink, upperPink)
            maskPink = cv2.morphologyEx(maskPink, cv2.MORPH_OPEN, kernel)
            maskPink = cv2.morphologyEx(maskPink, cv2.MORPH_CLOSE, kernel)

            cntsPink, heiP = cv2.findContours(maskPink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("Pink: " + str(cntsPink[0]))

            lowerGreen = np.array([green[0], green[1], green[2]])
            upperGreen = np.array([green[3], green[4], green[5]])
            maskGreen = cv2.inRange(hsvImg, lowerGreen, upperGreen)
            maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernel)
            maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_CLOSE, kernel)

            cntsGreen, heiP = cv2.findContours(maskGreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("Green: " + str(cntsGreen[0]))

            lowerBlue = np.array([blue[0], blue[1], blue[2]])
            upperBlue = np.array([blue[3], blue[4], blue[5]])
            maskBlue = cv2.inRange(hsvImg, lowerBlue, upperBlue)
            maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel)
            maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel)

            cntsBlue, heiP = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("Blue: " + str(cntsBlue[0]))

            lowerYellow = np.array([yellow[0], yellow[1], yellow[2]])
            upperYellow = np.array([yellow[3], yellow[4], yellow[5]])
            maskYellow = cv2.inRange(hsvImg, lowerYellow, upperYellow)
            maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel)
            maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel)

            cntsYellow, heiP = cv2.findContours(maskYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("Yellow: " + str(cntsYellow[0]))

            # pick most likely one (order by area?, largest is most likely)
            areas = []
            for c in cntsPink:
                # print(cv2.contourArea(c))
                areas.append(cv2.contourArea(c))
            # sort array based on areas
            sortedPink = bubble(areas, cntsPink)
            # sortedPink = [x for _, x in sorted(zip(areas, cntsPink))]
            # print(sortedPink)

            areas = []
            for c in cntsGreen:
                areas.append(cv2.contourArea(c))
            # sort array based on areas
            sortedGreen = bubble(areas, cntsGreen)
            # sortedGreen = [x for _, x in sorted(zip(areas, cntsGreen))]

            areas = []
            for c in cntsBlue:
                areas.append(cv2.contourArea(c))
            # sort array based on areas
            # sortedBlue = [x for _, x in sorted(zip(areas, cntsBlue))]
            sortedBlue = bubble(areas, cntsBlue)

            areas = []
            for c in cntsYellow:
                areas.append(cv2.contourArea(c))
            # sort array based on areas
            # sortedYellow = [x for _, x in sorted(zip(areas, cntsYellow))]
            sortedYellow = bubble(areas, cntsYellow)

            # origin, x, y, z
            sortedAll = [sortedGreen, sortedPink, sortedBlue, sortedYellow]

            # for each of the largest, find centerpoint
            cubePoints = np.zeros((4,2))
            landmarks = []
            for i in range(4):
                temp = sortedAll[i] # these are all contourse of a specific colour. Need to get center of 1st
                # print(np.shape(temp))
                M = cv2.moments(temp[0])
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cubePoints[i,:] = [cx, cy]
                pc = [float("NaN")]
                count = 0
                while np.isnan(pc).any() and count < 10000:
                    pc = self.getPoint(points, cx, cy)
                    count += 1
                    if count == 10000:
                        print (str(i) + " is nan")

                print(cx, cy)
                landmarks.append(pc)

                cv2.circle(colorImg, (cx, cy), radius=10, color=(0, 0, 255), thickness=3)
            print(landmarks)

            origin = np.array([0,0,0])
            ax1 = np.array([1,0,0])
            ax2 = np.array([0,1,0])
            ax3 = np.array([0,0,1])

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            o = landmarks[0]
            x = landmarks[1]
            y = landmarks[2]
            z = landmarks[3]

            dx = x - o
            dy = y - o
            dz = z - o
            dx = dx / np.linalg.norm(dx)
            dy = dy / np.linalg.norm(dy)
            dz = dz / np.linalg.norm(dz)

            landmarks = np.array([o, dx, dy, dz]) # essentially replace x, y, z
            R, t, angles = self.generateTransform(landmarks)
            print(angles)
            print(R)
            # print(t)
            # print()
            # print("Angles: " + str(angles))
            # points in robot coord sys
            # p1 = [0.5, 0, 0]
            # p2 = [0, 0.5, 0]
            # p3 = [0, 0, 0.5]

            p1 = [239, 357, 0]
            pc = [float("NaN")]
            count = 0
            while np.isnan(pc).any() and count < 10000:
                pc = self.getPoint(points, p1[0], p1[1])
                count += 1
                if count == 10000:
                    print("p1 is nan")
            p1 = pc

            p2 = [340, 355, 0]
            pc = [float("NaN")]
            count = 0
            while np.isnan(pc).any() and count < 10000:
                pc = self.getPoint(points, p2[0], p2[1])
                count += 1
                if count == 10000:
                    print("p2 is nan")
            p2 = pc

            p3 = [306, 289, 0]
            pc = [float("NaN")]
            count = 0
            while np.isnan(pc).any() and count < 10000:
                pc = self.getPoint(points, p3[0], p3[1])
                count += 1
                if count == 10000:
                    print("p3 is nan")
            p3 = pc

            # transform from robot to camera
            pc1 = self.cameraToRobot(p1, R, t)
            pc2 = self.cameraToRobot(p2, R, t)
            pc3 = self.cameraToRobot(p3, R, t)

            # do inverse camera to robot
            pr1 = self.robotToCamera(pc1, R, t)
            pr2 = self.robotToCamera(pc2, R, t)
            pr3 = self.robotToCamera(pc3, R, t)

            # print(p1, p2, p3)
            # print(pc1, pc2, pc3)
            # print(pr1, pr2, pr3)

            # self.plotCoords(ax, [origin, ax1, ax2, ax3])
            translatedLand = landmarks+o
            translatedLand[0] = o # gotta reset origin
            # self.plotCoords(ax, translatedLand)
            # self.plotPts(ax, [pc1, pc2, pc3])
            # self.plotPts(ax, [pr1, pr2, pr3])
            # plt.show()
            #self.show_image(colorImg)

            time.sleep(1)
            msg = Transform()
            msg.translation = Vector3(*t)
            quat = self.convertToQuat(R)
            print(quat)
            msg.rotation = Quaternion(*quat)
            pub.publish(msg)
            rate.sleep()
            rospy.signal_shutdown("Done!")


        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # except Exception as e:
        #     print(e)


rospy.init_node('callibration', anonymous=True)
test = Sub()

while not rospy.is_shutdown():
    rospy.spin()
