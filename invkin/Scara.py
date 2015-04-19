import numpy as np
from math import sqrt, cos, sin, acos, atan2

class Scara(object):
    "Kinematics and Inverse kinematics of a Scara (2dof planar arm)"

    def __init__(self, l1=1.0, l2=1.0, theta1=0.0, theta2=0.0):
        """
        Input:
        l1 - length of first link
        l2 - lenght of second link
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        """
        self.l1 = l1
        self.l2 = l2
        self.lsq = l1 ** 2 + l2 ** 2
        self.theta1 = theta1
        self.theta2 = theta2
        self.x, self.y = self.forward_kinematics()

    def update_joints(self, theta1, theta2):
        """
        Update the joint values
        Input:
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        Output:
        x, y - tool position in cartesian coordinates wrt arm base
        """
        self.theta1 = theta1
        self.theta2 = theta2
        self.x, self.y = self.forward_kinematics()

        return self.x, self.y

    def update_tool(self, x, y):
        """
        Update the tool position
        Input:
        x, y - tool position in cartesian coordinates wrt arm base
        Output:
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        """
        self.x = x
        self.y = y
        self.theta1, self.theta2 = self.inverse_kinematics()

        return self.theta1, self.theta2

    def forward_kinematics(self):
        """
        Computes tool position knowing joint positions
        """
        x = self.l1 * cos(self.theta1) + self.l2 * cos(self.theta1 + self.theta2)
        y = self.l1 * sin(self.theta1) + self.l2 * sin(self.theta1 + self.theta2)

        return x, y

    def inverse_kinematics(self):
        """
        Computes joint positions knowing tool position
        """
        l = self.x ** 2 + self.y ** 2
        lsq = self.lsq
        gamma = acos((l + self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * sqrt(l)))

        theta1 = atan2(self.y, self.x) - gamma
        theta2 = atan2(sqrt(1 - ((l - lsq) / (2 * self.l1 * self.l2)) ** 2), (l - lsq) / (2 * self.l1 * self.l2))

        return theta1, theta2
