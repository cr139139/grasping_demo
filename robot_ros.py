#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from actionlib import SimpleActionClient
from control_msgs.msg import GripperCommandAction, GripperCommandFeedback, GripperCommandResult, GripperCommandGoal

import numpy as np
from typing import List


class ROBOT(object):
    def __init__(self, bc):
        self.bc = bc
        self.robot_id: int = 0
        self.ee_id: int = 0
        self.arm_joint_ids: List[int] = [0]
        self.arm_rest_poses: List[float] = [0]
        self.gripper_link_ids: List[int] = [0]
        self.gripper_link_sign: List[float] = [1]
        self.gripper_link_limit: List[float] = [0, 1]
        self.arm_velocity: float = .35
        self.arm_force: float = 100.
        self.gripper_force: float = 20.

        self.ee_pos: List[float] = [0.537, 0.0, 0.5]
        self.ee_orn: List[float] = [0, -np.pi, 0]
        self.gripper_angle: float = 0

        rospy.init_node('talker', anonymous=True)
        # self.arm_command = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=100)
        self.arm_command = rospy.Publisher('/custom/command', Float64MultiArray, queue_size=100)
        self.gripper_command = SimpleActionClient('/ezgripper_single_mount/ezgripper_controller/gripper_cmd', GripperCommandAction)
        self.gripper_command.wait_for_server()

        self.arm_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self.callback_joint)
        self.gripper_sub = rospy.Subscriber("/ezgripper_single_mount/joint_states", JointState, self.callback_gripper)

        self.ros_msg = Float64MultiArray()
        self.arm_joint_states = np.zeros(7)
        self.gripper_state = 0

    def callback_joint(self, msg):
        self.arm_joint_states = np.array(msg.position)

    def callback_gripper(self, msg):
        self.gripper_state = np.array(msg.position)

    def gripper_constraint(self):
        for i in range(len(self.gripper_link_ids)):
            if i != 0:
                c = self.bc.createConstraint(self.robot_id, self.gripper_link_ids[0],
                                             self.robot_id, self.gripper_link_ids[i],
                                             jointType=self.bc.JOINT_GEAR,
                                             jointAxis=[0, 1, 0],
                                             parentFramePosition=[0, 0, 0],
                                             childFramePosition=[0, 0, 0])
                gearRatio = -self.gripper_link_sign[0] * self.gripper_link_sign[i]
                self.bc.changeConstraint(c, gearRatio=gearRatio, maxForce=3, erp=1)

            gripper_link_limit = sorted([limit * self.gripper_link_sign[i] for limit in self.gripper_link_limit])
            self.bc.changeDynamics(self.robot_id, self.gripper_link_ids[i],
                                   jointLowerLimit=gripper_link_limit[0],
                                   jointUpperLimit=gripper_link_limit[1])

    def reset_arm_poses(self):
        self.ros_msg.data = self.arm_rest_poses[:7]
        self.arm_command.publish(self.ros_msg)
        for rest_pose, joint_id in zip(self.arm_joint_states, self.arm_joint_ids):
            self.bc.resetJointState(self.robot_id, joint_id, rest_pose)
            self.bc.setJointMotorControl2(self.robot_id, joint_id, self.bc.POSITION_CONTROL,
                                          targetPosition=rest_pose, force=self.arm_force)

    def reset_gripper(self):
        gripper_goal = GripperCommandGoal()
        gripper_goal.command.max_effort = 10.0
        gripper_goal.command.position = self.gripper_link_limit[1]

        self.gripper_command.send_goal(gripper_goal)
        self.gripper_command.wait_for_result()
        self.gripper_command.get_result()

        for i in range(len(self.gripper_link_ids)):
            self.bc.resetJointState(self.robot_id, self.gripper_link_ids[i],
                                    self.gripper_link_limit[1] * self.gripper_link_sign[i])

    def control_gripper(self, position):
        position = np.clip(position, self.gripper_link_limit[0], self.gripper_link_limit[1])
        gripper_goal = GripperCommandGoal()
        gripper_goal.command.max_effort = 10.0
        gripper_goal.command.position = position

        self.gripper_command.send_goal(gripper_goal)
        self.gripper_command.wait_for_result()
        self.gripper_command.get_result()

        self.bc.setJointMotorControlArray(self.robot_id, self.gripper_link_ids, self.bc.POSITION_CONTROL,
                                          targetPositions=[i * self.gripper_state for i in self.gripper_link_sign],
                                          positionGains=[1 for _ in range(len(self.gripper_link_ids))],
                                          forces=[self.gripper_force for _ in range(len(self.gripper_link_ids))])

    def control_arm(self, positions):
        self.ros_msg.data = positions[:7]
        self.arm_command.publish(self.ros_msg)
        for position, joint_id in zip(self.arm_joint_states, self.arm_joint_ids):
            self.bc.setJointMotorControl2(self.robot_id, joint_id, self.bc.POSITION_CONTROL,
                                          targetPosition=position, force=self.arm_force)

    def applyAction(self, motor_commands):
        self.ee_pos = np.array(self.ee_pos) + np.array(motor_commands[:3])
        self.ee_pos[0] = np.clip(self.ee_pos[0], 0.50, 0.65)
        self.ee_pos[1] = np.clip(self.ee_pos[1], -0.17, 0.22)
        self.ee_orn[2] += motor_commands[3]
        self.gripper_angle -= motor_commands[4]

        ee_orn_quaternion = self.bc.getQuaternionFromEuler(self.ee_orn)
        joint_poses = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
                                                         self.ee_pos, ee_orn_quaternion)
        self.control_arm(joint_poses)
        self.control_gripper(self.gripper_angle)

    def get_joint_limits(self, body_id, joint_ids):
        """Query joint limits as (lo, hi) tuple, each with length same as
        `joint_ids`."""
        joint_limits = []
        for joint_id in joint_ids:
            joint_info = self.bc.getJointInfo(body_id, joint_id)
            joint_limit = joint_info[8], joint_info[9]
            joint_limits.append(joint_limit)
        joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
        return joint_limits


class KUKASAKE(ROBOT):
    def __init__(self, bc, pos, orn):
        super().__init__(bc)
        self.robot_id: int = bc.loadURDF('/robots/iiwa7_sake.urdf', pos, orn,
                                         useFixedBase=True, flags=bc.URDF_USE_SELF_COLLISION)

        self.ee_id: int = 7  # NOTE(choi): End-effector joint ID for UR5 robot
        self.arm_joint_ids: List[int] = [0, 1, 2, 3, 4, 5, 6]  # NOTE(choi): Hardcoded arm joint id for UR5 robot
        self.arm_joint_limits = self.get_joint_limits(self.robot_id, self.arm_joint_ids)
        self.arm_rest_poses: List[float] = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
                                                                              self.ee_pos,
                                                                              self.bc.getQuaternionFromEuler(
                                                                                  self.ee_orn),
                                                                              maxNumIterations=100)
        self.gripper_z_offset: float = 0.25
        self.gripper_link_ids: List[int] = [10, 11, 13, 14]
        self.gripper_link_sign: List[float] = [-1, 0, -1, 0]
        self.gripper_link_limit: List[float] = [-0.27, 1.57075]
        self.gripper_angle: float = self.gripper_link_limit[1]
        self.gripper_constraint()

        self.reset_arm_poses()
        self.reset_gripper()
