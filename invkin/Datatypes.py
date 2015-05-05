from collections import namedtuple
from math import pi

FLIP_RIGHT_HAND = 1
FLIP_LEFT_HAND = -1
EPSILON = 1e-2
GRIPPER_ANGULAR_SPACING = 2 * pi / 3

RobotSpacePoint = namedtuple('RobotSpacePoint', ['x', 'y', 'z', 'gripper_hdg'])
JointSpacePoint = namedtuple('JointSpacePoint', ['theta1', 'theta2', 'z', 'theta3'])

Vector2D = namedtuple('Vector2D', ['x', 'y'])
