#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Transform


def callback(data):
    t = data.translation
    rospy.loginfo(rospy.get_caller_id() + "translation: %s", t)
    r = data.rotation
    rospy.loginfo(rospy.get_caller_id() + "quaternion: %s", r)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("transformation", Transform, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
