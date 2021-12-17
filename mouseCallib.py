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
from pprint import pprint


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
        t = o

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
        # return pList[0]
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



    def image_callback(self, image1, image2, points):
        global o, x, y, z, counter, colorImg, mousex, mousey, flag, landmarks
        try:

            colorImg = bridge.imgmsg_to_cv2(image1, "bgr8")
            if counter == 0:
                cv2.putText(colorImg, "Please click on the origin point", (200, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            elif counter == 1:
                cv2.putText(colorImg, "Please click on the x point", (200, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            elif counter == 2:
                cv2.putText(colorImg, "Please click on the y point", (200, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            elif counter == 3:
                cv2.putText(colorImg, "Please click on the z point", (200, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)


            if flag == 1:
                # using mousex and mousey, get world coordinates
                pc = self.getPoint(points, mousex, mousey)
                landmarks.append(pc)
                flag = 0

            if counter == 4:
                cv2.destroyAllWindows()
                # print(landmarks)
                # print()
                # get transformation
                #landmarks = np.array([[0.0441, 0.3042, 0.995],[0.129, 0.3060, 1.047],[-0.0125, 0.2969, 1.065],[0.049, 0.162, 1.006]])


                origin = [0,0,0]
                ax1 = [1,0,0]
                ax2 = [0,1,0]
                ax3 = [0,0,1]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # ax.set_aspect('equal')
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

                landmarks = np.array([o, dx, dy, dz])
                R, t, angles = self.generateTransform(landmarks)

                # print("Angles: " + str(angles))
                # points in robot coord sys
                p1 = [0.5, 0, 0]
                p2 = [0, 0.5, 0]
                p3 = [0, 0, 0.5]

                # transform from robot to camera
                pc1 = self.robotToCamera(p1, R, t)
                pc2 = self.robotToCamera(p2, R, t)
                pc3 = self.robotToCamera(p3, R, t)

                # do inverse camera to robot
                pr1 = self.cameraToRobot(pc1, R, t)
                pr2 = self.cameraToRobot(pc2, R, t)
                pr3 = self.cameraToRobot(pc3, R, t)

                # print(p1, p2, p3)
                # print(pc1, pc2, pc3)
                # print(pr1, pr2, pr3)

                self.plotCoords(ax, [origin, ax1, ax2, ax3])
                translatedLand = landmarks+o
                translatedLand[0] = o # gotta reset origin
                self.plotCoords(ax, translatedLand)
                self.plotPts(ax, [pc1, pc2, pc3])
                self.plotPts(ax, [pr1, pr2, pr3])
                plt.show()

                rospy.signal_shutdown("Done!")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        self.show_image(colorImg)

rospy.init_node('pillarTracker', anonymous=True)
test = Sub()

while not rospy.is_shutdown():
    rospy.spin()
