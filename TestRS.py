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

bridge = CvBridge()

class Sub:
    def __init__(self):

        colorTopic = '/camera/color/image_raw'
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
            colorImg = bridge.imgmsg_to_cv2(image1, "bgr8")

            image2.encoding = "mono16"
            depth = bridge.imgmsg_to_cv2(image2, desired_encoding='mono8')
            depthImg = np.array(depth, dtype=np.float32)



            # for color image, remove everything beyond threshold
            gray = 153
            depth3D = np.dstack((depthImg, depthImg, depthImg))
            thresh = 7
            bg_removed = np.where((depth3D > thresh) | (depth3D <= 0), gray, colorImg)


            x, y, = 300, 250

            # within bounding box, find average depth data with outliers removed, if within range of arm (1m?) calculate size
            #print(int(x+w/2), int(y+ h/2))
            # print(depthImg.shape)
            # print(colorImg.shape)
            depth = depthImg[x][y]
            #print(temp)
            cv2.putText(bg_removed, "(" + str(x) + ", " + str(y) + ", " + str(depth) + ")", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)



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