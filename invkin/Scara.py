from invkin.Datatypes import *
from math import sqrt, cos, sin, acos, atan2, pi

class Scara(object):
    "Kinematics and Inverse kinematics of a Scara (2dof planar arm)"

    def __init__(self, l1=1.0, l2=1.0, q0=JointSpacePoint(0,0,0,0), \
                 origin=Vector2D(0,0), flip_x=FLIP_RIGHT_HAND):
        """
        Input:
        l1 - length of first link
        l2 - lenght of second link
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        flip_x - vertical flip (positive for right hand, negative for left hand)
        """
        self.l1 = l1
        self.l2 = l2
        self.lsq = l1 ** 2 + l2 ** 2
        self.joints = q0
        self.origin = origin

        if flip_x >= 0:
            self.flip_x = FLIP_RIGHT_HAND
        else:
            self.flip_x = FLIP_LEFT_HAND

        self.tool = self.forward_kinematics()

    def update_joints(self, new_joints):
        """
        Update the joint values
        Input:
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        Output:
        x, y - tool position in cartesian coordinates wrt arm base
        """
        self.joints = new_joints
        self.tool = self.forward_kinematics()

        return self.tool

    def update_tool(self, new_tool):
        """
        Update the tool position
        Input:
        x, y - tool position in cartesian coordinates wrt arm base
        Output:
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        """
        norm = (new_tool.x - self.origin.x) ** 2 + (new_tool.y - self.origin.y) ** 2
        if(norm > (self.l1 + self.l2) ** 2):
            "Target unreachable"
            x = self.flip_x * (self.l1 + self.l2) + self.origin.x
            y = 0 + self.origin.y
            self.tool = RobotSpacePoint(x, y, 0, 0)
            self.joints = self.inverse_kinematics()
            raise ValueError('Target unreachable')

        self.tool = new_tool
        self.joints = self.inverse_kinematics()

        return self.joints

    def forward_kinematics(self):
        """
        Computes tool position knowing joint positions
        """
        x = self.flip_x * (self.l1 * cos(self.joints.theta1) \
            + self.l2 * cos(self.joints.theta1 + self.joints.theta2))
        y = self.l1 * sin(self.joints.theta1) \
            + self.l2 * sin(self.joints.theta1 + self.joints.theta2)

        x += self.origin.x
        y += self.origin.y

        return RobotSpacePoint(x, y, 0, 0)

    def inverse_kinematics(self):
        """
        Computes joint positions knowing tool position
        """
        x = self.tool.x - self.origin.x
        y = self.tool.y - self.origin.y

        if(x == 0 and y == 0):
            return JointSpacePoint(self.joints.theta1, pi, 0, 0)

        l = x ** 2 + y ** 2
        lsq = self.lsq

        cos_gamma = (l + self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * sqrt(l))

        # Numerical errors can make abs(cos_gamma) > 1
        if(cos_gamma > 1 - EPSILON or cos_gamma < -1 + EPSILON):
            gamma = 0.0
        else:
            gamma = acos(cos_gamma)

        theta1 = atan2(y, self.flip_x * x) - gamma
        theta2 = atan2(sqrt(1 - ((l - lsq) / (2 * self.l1 * self.l2)) ** 2), \
                            (l - lsq) / (2 * self.l1 * self.l2))

        return JointSpacePoint(theta1, theta2, 0, 0)

    def get_detailed_pos(self):
        """
        Returns origin_x, origin_y, x1, y1, x2, y2
        """
        x1 = self.flip_x * self.l1 * cos(self.joints.theta1) + self.origin.x
        y1 = self.l1 * sin(self.joints.theta1) + self.origin.y

        return self.origin, Vector2D(x1, y1), Vector2D(self.tool.x, self.tool.y)
