from invkin.Datatypes import *
from invkin.Scara import Scara
from math import pi, cos, sin

class DebraArm(Scara):
    "Kinematics and Inverse kinematics of an arm on Debra (3dof + hand)"

    def __init__(self, l1=1.0, l2=1.0, q0=JointSpacePoint(0,0,0,0), \
                 origin=Vector3D(0,0,0), flip_x=FLIP_RIGHT_HAND):
        """
        Input:
        l1 - length of first link
        l2 - lenght of second link
        q0 - initial positions of joints
        origin - position of the base of the arm in carthesian space
        flip_x - vertical flip (positive for right hand, negative for left hand)
        """
        self.l1 = l1
        self.l2 = l2
        self.lsq = l1 ** 2 + l2 ** 2
        self.joints = q0
        self.origin = origin

        if flip_x > 0:
            self.flip_x = FLIP_RIGHT_HAND
        else:
            self.flip_x = FLIP_LEFT_HAND

        self.tool = self.forward_kinematics()

    def update_joints(self, new_joints):
        """
        Update the joint values
        Input:
        new_joints - new positions of joints
        Output:
        tool - tool position in cartesian coordinates wrt arm base
        """
        self.joints = new_joints
        self.tool = self.forward_kinematics()

        return self.tool

    def update_tool(self, new_tool):
        """
        Update the tool position
        Input:
        new_tool - tool position in cartesian coordinates wrt arm base
        Output:
        new_joints - position of joints
        """
        norm = (new_tool.x - self.origin.x) ** 2 + (new_tool.y - self.origin.y) ** 2
        if(norm > (self.l1 + self.l2) ** 2 or norm < (self.l1 - self.l2) ** 2):
            "Target unreachable"
            self.tool = self.forward_kinematics()
            raise ValueError('Target unreachable')

        self.tool = new_tool
        self.joints = self.inverse_kinematics()

        return self.joints

    def forward_kinematics(self):
        """
        Computes tool position knowing joint positions
        """
        tool = super(DebraArm, self).forward_kinematics()

        grp_hdg = (pi / 2) - (self.joints.theta1 + self.joints.theta2 \
                                                 + self.joints.theta3)
        tool = tool._replace(z=(self.joints.z + self.origin.z), gripper_hdg=grp_hdg)

        return tool

    def inverse_kinematics(self):
        """
        Computes joint positions knowing tool position
        """
        x = self.tool.x - self.origin.x
        y = self.tool.y - self.origin.y
        z = self.tool.z - self.origin.z
        gripper_hdg = self.tool.gripper_hdg

        joints = super(DebraArm, self).inverse_kinematics()

        th3 = pi / 2 - (gripper_hdg + joints.theta1 + joints.theta2)
        joints = joints._replace(z=z, theta3=th3)

        return joints

    def get_detailed_pos(self, l3):
        """
        Returns origin, p1, p2, p3, z
        """
        p0, p1, p2 = super(DebraArm, self).get_detailed_pos()

        x3 = self.tool.x + self.flip_x * l3 * cos(self.joints.theta1 \
                                                  + self.joints.theta2 \
                                                  + self.joints.theta3)
        y3 = self.tool.y + l3 * sin(self.joints.theta1 \
                                    + self.joints.theta2 \
                                    + self.joints.theta3)

        return self.origin, p1, p2, Vector2D(x3, y3), self.tool.z
