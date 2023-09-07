#!/usr/bin/env python
import rospy
import sys
import termios
import tty
from select import select

def getKey(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    # sys.stdin.read() returns a string on Linux
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    while not rospy.is_shutdown():
        settings = saveTerminalSettings()
        key = getKey(settings, 0.001)
        restoreTerminalSettings(settings)
        if key:
            print(key)
        if key == "q":
            break
