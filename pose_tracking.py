#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from actionlib import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandFeedback, GripperCommandResult, GripperCommandGoal

import time
import numpy as np
from optitrack_ros import OPTITRACKROS, quat_to_rot, rot_to_quat

ee_state = Pose()
gripper_state = 0


def angular_distance(q1, q2):
    R1 = quat_to_rot(q1)
    R2 = quat_to_rot(q2)
    return 1 - rot_to_quat(R1.T @ R2)[3]
    return np.linalg.norm(q1[3] * q2[0:3] + q2[3] * q1[0:3] + np.cross(q1[:3], q2[:3]))


def callback_ee(msg):
    global ee_state
    ee_state = msg


def callback_gripper(msg):
    global gripper_state
    gripper_state = np.array(msg.position)


rospy.init_node('talker', anonymous=True)
ee_pub = rospy.Publisher('/iiwa/desired_pos', Pose, queue_size=100)
gripper_pub = SimpleActionClient('/ezgripper_single_mount/ezgripper_controller/gripper_cmd', GripperCommandAction)
gripper_pub.wait_for_server()
ee_sub = rospy.Subscriber("/iiwa/task_states", Pose, callback_ee)
gripper_sub = rospy.Subscriber("/ezgripper_single_mount/joint_states", JointState, callback_gripper)
optitrack = OPTITRACKROS(['kuka', 'kukaObject'])
rate = rospy.Rate(100)
time.sleep(5)

ee_msg = ee_state
gripper_goal = GripperCommandGoal()
gripper_goal.command.max_effort = 10.0

# in Optitrack frame
R_init = np.array([[0, -1, 0],
                   [0, 0, 1],
                   [-1, 0, 0]])
z_offset = np.array([0, -0.24, 0])
grasp_status = False

home_pos = True

while not rospy.is_shutdown():
    poses = optitrack.reading()
    robot_pose = poses['robot']
    object_pose = poses['object']

    if home_pos:
        gripper_pub.send_goal(1.57075)
        gripper_pub.wait_for_result()
        gripper_pub.get_result()

        ee_msg.position = Point(0.6, 0, 0.53433)
        ee_msg.orientation = Quaternion(0, 0, 0, 1)
        ee_pub.publish(ee_msg)
    else:
        # in Optitrack frame
        R_diff = quat_to_rot(robot_pose[1]).T @ quat_to_rot(object_pose[1]) @ R_init
        q_diff = rot_to_quat(R_diff)
        t_diff = object_pose[0] - robot_pose[0]
        t_diff += quat_to_rot(object_pose[1]) @ z_offset

        # in ROS frame
        t_diff = np.array([t_diff[1], -t_diff[0], t_diff[2]])
        q_diff = np.array([-q_diff[2], -q_diff[0], -q_diff[1], q_diff[3]])

        # gripper DS
        t_ee = np.array([ee_state.position.x, ee_state.position.y, ee_state.position.z])
        q_ee = np.array([ee_state.orientation.x, ee_state.orientation.y, ee_state.orientation.z, ee_state.orientation.w])
        t_dist = np.max([np.linalg.norm(t_diff - t_ee) - 0.04, 0])
        q_dist = angular_distance(q_diff, q_ee)

        alpha = 1
        beta = 30
        gripper_goal_dot = -(gripper_state + 0.27 - beta * (0.1 * t_dist + 0.9 * q_dist))
        gripper_goal.command.position = np.clip(gripper_state + alpha * gripper_goal_dot, -0.27, 1.57075)
        print(t_dist, q_dist)
        if gripper_goal.command.position < -0.1:
            if not grasp_status:
                t_goal = t_ee
            gripper_goal.command.position = -0.27
            grasp_status = True
        else:
            t_goal = t_diff
            grasp_status = False

        gripper_pub.send_goal(gripper_goal)
        gripper_pub.wait_for_result()
        gripper_pub.get_result()

        ee_msg.position = Point(t_goal[0], t_goal[1], t_goal[2])
        ee_msg.orientation = Quaternion(q_diff[0], q_diff[1], q_diff[2], q_diff[3])
        ee_pub.publish(ee_msg)
    rate.sleep()
