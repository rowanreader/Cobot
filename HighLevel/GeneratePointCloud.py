# saves pointcloud when octomap runs
import rospy
import rosbag
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Point

# def callback(data):
#     point = [data.x, data.y, data.z]
#     bag.write('tileCloud', point)
#     # rospy.loginfo(rospy.get_caller_id() + "I heard %s", point)
#
#
# def listener():
#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     rospy.init_node('listener', anonymous=True)
#
#     rospy.Subscriber("centerpoints", Point, callback)
#
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()

bag = rosbag.Bag('tile10.bag', 'w')

try:
    s = String()
    s.data = 'foo'

    i = Int32()
    i.data = 42

    bag.write('chatter', s)
    bag.write('numbers', i)
finally:
    bag.close()