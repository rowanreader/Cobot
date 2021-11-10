#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point


def callback(data):
    point = [data.x, data.y, data.z]
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", point)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("centerpoints", Point, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()