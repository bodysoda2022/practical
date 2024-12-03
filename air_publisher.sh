#!/bin/bash

mkdir ~/catkin_ws

cd catkin_ws

mkdir src

cd src

catkin_create_pkg my_robot_description std_msgs rospy roscpp

cd my_robot_description

mkdir scripts

publisher_code="#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        hello_str = \"hello world %s\" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass"

echo "$publisher_code" > scripts/publisher.py

subscriber_code="#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
def callback(data):
    rospy.loginfo(\"I heard %s\", data.data)
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', String, callback)
    rospy.spin()
if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass"

echo "$subscriber_code" > scripts/subscriber.pychmod +x scripts/publisher.py scripts/subscriber.py

mkdir launch

launch_file_content="<launch>
  <!-- Launch the publisher node -->
  <node name=\"publisher_node\" pkg=\"my_robot_description\" type=\"publisher.py\" output=\"screen\" />

  <!-- Launch the subscriber node -->
  <node name=\"subscriber_node\" pkg=\"my_robot_description\" type=\"subscriber.py\" output=\"screen\" />
</launch>"

echo "$launch_file_content" > launch/my_launch_file.launch

cd ~/catkin_ws

catkin_make

source devel/setup.bash

roslaunch my_robot_description my_launch_file.launch
