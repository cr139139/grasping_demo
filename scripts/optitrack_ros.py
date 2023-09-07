#! /usr/bin/env python3
import rospy
import tf2_ros
import numpy as np


def rot_to_quat(R):
    t = np.matrix.trace(R)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if t > 0:
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[2, 1] - R[1, 2]) * t
        q[1] = (R[0, 2] - R[2, 0]) * t
        q[2] = (R[1, 0] - R[0, 1]) * t

    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (R[k, j] - R[j, k]) * t
        q[j] = (R[j, i] + R[i, j]) * t
        q[k] = (R[k, i] + R[i, k]) * t

    return q


def quat_to_rot(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def tf2_list(tf2):
    return [np.array([tf2.transform.translation.x,
                      tf2.transform.translation.y,
                      tf2.transform.translation.z]),
            np.array([tf2.transform.rotation.x,
                      tf2.transform.rotation.y,
                      tf2.transform.rotation.z,
                      tf2.transform.rotation.w])]


class OPTITRACKROS:
    def __init__(self, tf_list):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_list = tf_list

    def reading(self):
        readings = {}
        try:
            for tf in self.tf_list:
                readings[tf] = tf2_list(self.tfBuffer.lookup_transform('world', tf, rospy.Time(0)))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        return readings
