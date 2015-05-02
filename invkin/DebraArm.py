from invkin.Scara import Scara
from math import pi

GRIPPER_ANGULAR_SPACING = 2 * pi / 3
FLIP_RIGHT_HAND = 1
FLIP_LEFT_HAND = -1
EPSILON = 1e-2

class DebraArm(Scara):
    "Kinematics and Inverse kinematics of an arm on Debra (3dof + hand)"

    def __init__(self, l1=1.0, l2=1.0, theta1=0.0, theta2=0.0, z=0.0, theta3=0.0, \
                 origin=(0.0, 0.0, 0.0), flip_x=FLIP_RIGHT_HAND):
        """
        Input:
        l1 - length of first link
        l2 - lenght of second link
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        z - position on z axis
        theta3 - angle of the hand wrt to the second link
        gripper_hdg - heading of the hand wrt heading of robot
        flip_x - vertical flip (positive for right hand, negative for left hand)
        """
        self.l1 = l1
        self.l2 = l2
        self.lsq = l1 ** 2 + l2 ** 2
        self.theta1 = theta1
        self.theta2 = theta2
        self.z = z
        self.theta3 = theta3
        self.origin = origin

        if flip_x > 0:
            self.flip_x = FLIP_RIGHT_HAND
        else:
            self.flip_x = FLIP_LEFT_HAND

        self.x, self.y, self.z, self.gripper_hdg = self.forward_kinematics()

    def update_joints(self, theta1, theta2, z, theta3):
        """
        Update the joint values
        Input:
        theta1 - angle of the first link wrt ground
        theta2 - angle of the second link wrt the first
        z - position on z axis
        theta3 - angle of the hand wrt to the second link
        Output:
        x, y, z - tool position in cartesian coordinates wrt arm base
        gripper_hdg - heading of the hand to use wrt heading of robot
        """
        self.theta1 = theta1
        self.theta2 = theta2
        self.z = z
        self.theta3 = theta3

        self.x, self.y, self.z, self.gripper_hdg = self.forward_kinematics()

        return self.x + self.origin[0], \
               self.y + self.origin[1], \
               self.z + self.origin[2], \
               self.gripper_hdg

    def forward_kinematics(self):
        """
        Computes tool position knowing joint positions
        """
        x, y = super(DebraArm, self).forward_kinematics()
        z = self.z

        gripper_hdg = (pi / 2) - (self.theta1 + self.theta2 + self.theta3)

        return x, y, z, gripper_hdg
