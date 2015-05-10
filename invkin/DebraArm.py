from invkin.Datatypes import *
from invkin.Constraints import Constraints
from invkin.Scara import Scara
from math import pi, cos, sin
import numpy as np

class DebraArm(Scara):
    "Kinematics and Inverse kinematics of an arm on Debra (3dof + hand)"

    def __init__(self, l1=1.0, l2=1.0, constraints=Constraints(),
                 q0=JointSpacePoint(0,0,0,0),
                 origin=Vector3D(0,0,0),
                 flip_x=FLIP_RIGHT_HAND):
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
        self.constraints = constraints
        self.joints = q0
        self.origin = origin

        if flip_x > 0:
            self.flip_x = FLIP_RIGHT_HAND
        else:
            self.flip_x = FLIP_LEFT_HAND

        self.flip_elbow = ELBOW_BACK

        self.tool = self.get_tool()

    def forward_kinematics(self, new_joints):
        """
        Update the joint values through computation of forward kinematics
        """
        self.joints = new_joints
        self.tool = self.get_tool()

        return self.tool

    def inverse_kinematics(self, new_tool):
        """
        Update the tool position through computation of inverse kinematics
        """
        norm = (new_tool.x - self.origin.x) ** 2 + (new_tool.y - self.origin.y) ** 2
        if(norm > (self.l1 + self.l2) ** 2 or norm < (self.l1 - self.l2) ** 2):
            "Target unreachable"
            self.tool = self.get_tool()
            raise ValueError('Target unreachable')

        self.tool = new_tool
        self.joints = self.get_joints()

        return self.joints

    def get_tool(self):
        """
        Computes tool position knowing joint positions
        """
        tool = super(DebraArm, self).get_tool()

        grp_hdg = (pi / 2) - (self.joints.theta1 \
                              + self.joints.theta2 \
                              + self.joints.theta3)
        tool = tool._replace(z=(self.joints.z + self.origin.z), gripper_hdg=grp_hdg)

        return tool

    def get_joints(self):
        """
        Computes joint positions knowing tool position
        """
        x = self.tool.x - self.origin.x
        y = self.tool.y - self.origin.y
        z = self.tool.z - self.origin.z
        gripper_hdg = self.tool.gripper_hdg

        joints = super(DebraArm, self).get_joints()

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

    def compute_jacobian(self):
        """
        Returns jacobian matrix at current state
        """
        dx_dth1 = - self.l1 * sin(self.joints.theta1) \
                  - self.l2 * sin(self.joints.theta1 + self.joints.theta2)
        dx_dth2 = - self.l2 * sin(self.joints.theta1 + self.joints.theta2)

        dy_dth1 = self.l1 * cos(self.joints.theta1) \
                  + self.l2 * cos(self.joints.theta1 + self.joints.theta2)
        dy_dth2 = self.l2 * cos(self.joints.theta1 + self.joints.theta2)

        return np.matrix([[dx_dth1, dx_dth2, 0,  0], \
                          [dy_dth1, dy_dth2, 0,  0], \
                          [      0,       0, 1,  0], \
                          [     -1,      -1, 0, -1]])

    def get_tool_vel(self, joints_vel):
        """
        Computes current tool velocity using jacobian
        """
        joints_vel = np.matrix([[joints_vel.theta1], \
                                [joints_vel.theta2], \
                                [joints_vel.z], \
                                [joints_vel.theta3]])
        jacobian = self.compute_jacobian()
        tool_vel = jacobian * joints_vel

        return RobotSpacePoint(tool_vel[0], \
                               tool_vel[1], \
                               tool_vel[2], \
                               tool_vel[3])

    def get_joints_vel(self, tool_vel):
        """
        Computes current tool velocity using jacobian
        """
        tool_vel = np.matrix([[tool_vel.x], \
                              [tool_vel.y], \
                              [tool_vel.z], \
                              [tool_vel.gripper_hdg]])
        jacobian = self.compute_jacobian()

        if abs(np.linalg.det(jacobian)) < EPSILON:
            raise ValueError('Singularity')

        joints_vel = np.linalg.solve(jacobian, tool_vel)

        return JointSpacePoint(joints_vel[0], \
                               joints_vel[1], \
                               joints_vel[2], \
                               joints_vel[3])
