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
from roslib import message
from pprint import pprint

bridge = CvBridge()

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

    def show_image(self, img):
        cv2.imshow("Image Window", img)
        cv2.waitKey(3)

    # def dimage_callback(img_msg):
    #     depth = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    #     depthImg = np.array(depth, dtype=np.float32)
    #     return depthImg

    def getPoint(self, points, x, y):
        # x goes up to 640, y goes up to 480
        pc1 = pc2.read_points(points, skip_nans=False, field_names=("x", "y", "z"), uvs=[(x, y)])
        p = list(pc1)[0]
        return p

    def image_callback(self, image1, image2, points):
        try:
            colorImg = bridge.imgmsg_to_cv2(image1, "bgr8")

            image2.encoding = "mono16"
            depth = bridge.imgmsg_to_cv2(image2, desired_encoding='mono16')
            depthImg = np.array(depth, dtype=np.float32)

            # each point has 4 values: x, y ,z, i (i being intensity, representing accuracy?)
            # pc = ros_numpy.numpify(points)
            pc = pc2.read_points(points, skip_nans=False, field_names=("x", "y", "z"))
            # pc1 = pc2.read_points(points, skip_nans=False, field_names = ("x", "y", "z"), uvs=[(500, 0)])

            pList = []
            for p in pc:
                pList.append([p[0], p[1], p[2]])


            # points = np.zeros((len(pc), 3))
            # points[:, 0] = pc['x']
            # points[:, 1] = pc['y']
            # points[:, 2] = pc['z']
            # #p = pcl.PointCloud2(np.array(points, dtype=np.float32))
            # print(pList[500])
            # print(list(pc1)[0])
            # print(" ")
            #pprint(vars(pc))
            # pprint(np.shape(pList))
            #print(points.data)
            # pprint(np.shape(points))
            # pprint(np.shape(pc))
            # print(" ")

            #pointCloud = bridge.imgmsg_to_cv2(points)
            #print(np.shape(pointCloud))

            # for color image, remove everything beyond threshold
            gray = 153
            depth3D = np.dstack((depthImg, depthImg, depthImg))
            thresh = 1800
            # bg_removed = np.where((depth3D > thresh) | (depth3D <= 0), gray, colorImg)
            bg_removed = colorImg
            # 15.2 25.1
            x, y = 350, 379
            p = np.round(self.getPoint(points, x, y),4)
            #print(p)

            depth = depthImg[y][x]
            #print(temp)
            cv2.putText(bg_removed, "(" + str(p[0]) + ", " + str(p[1]) + ", " + str(p[2]) + ")", (x-200, y+50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(bg_removed, "(" + str(x) + ", " + str(y) + ", " + str(np.round(depth/100,3)) + ")", (x-200, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(bg_removed, (x, y), radius=2, color=(0, 0, 255), thickness=-1)


        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))



        norm_image = cv2.normalize(depthImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(bg_removed)
        #gray = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        self.show_image(bg_removed)

rospy.init_node('pillarTracker', anonymous=True)
test = Sub()

# k = cv2.waitKey(1)
#     if k == 27:
#         break
#
while not rospy.is_shutdown():
    rospy.spin()
