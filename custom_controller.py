#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist, Vector3
import numpy as np
import time

joint_position = np.zeros(7)
joint_velocity = np.zeros(7)
joint_target = np.zeros(7)


def callback_joint(msg):
    global joint_position
    joint_position = np.array(msg.position)


def callback_target(msg):
    global joint_target
    joint_target = np.array(msg.data)


pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=100)
sub = rospy.Subscriber("/iiwa/joint_states", JointState, callback_joint)
tar = rospy.Subscriber("/custom/command", Float64MultiArray, callback_target)

rospy.init_node('pose_tracker', anonymous=True)
ros_msg = Float64MultiArray()
rate = rospy.Rate(500)

time.sleep(1)
joint_target = joint_position
joint_target = np.zeros(7)
while True:
    joint_velocity = (joint_target - joint_position) * 10
    ros_msg.data = joint_position + joint_velocity * 1 / 500.
    pub.publish(ros_msg)
    rate.sleep()
