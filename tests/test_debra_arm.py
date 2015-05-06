from invkin.Datatypes import *
from invkin import DebraArm
from math import pi, sqrt, cos, sin
import numpy as np
import unittest

l1 = 1.0
l2 = 0.5

class DebraArmTestCase(unittest.TestCase):
    def test_fwdkin(self):
        """
        Checks that forward kinematics works
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        th1 = pi / 2
        th2 = pi / 2
        z = 0.1
        th3 = pi / 2
        joints = JointSpacePoint(th1, th2, z, th3)
        tool = arm.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, -l2)
        self.assertAlmostEqual(tool.y, l1)
        self.assertAlmostEqual(tool.z, 0.1)
        self.assertAlmostEqual(tool.gripper_hdg, -pi)

        th1 = -pi / 2
        th2 = pi / 2
        z = 0.2
        th3 = pi / 2
        joints = JointSpacePoint(th1, th2, z, th3)
        tool = arm.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l2)
        self.assertAlmostEqual(tool.y, -l1)
        self.assertAlmostEqual(tool.z, 0.2)
        self.assertAlmostEqual(tool.gripper_hdg, 0.0)

    def test_invkin(self):
        """
        Checks that inverse kinematics works
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        x = l2
        y = - l1
        z = 0.1
        grp_hdg = 0
        tool = RobotSpacePoint(x, y, z, grp_hdg)
        joints = arm.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)
        self.assertAlmostEqual(joints.z, 0.1)
        self.assertAlmostEqual(joints.theta3, pi / 2)

        x = 0
        y = l1 + l2
        z = 0.2
        grp_hdg = 0
        tool = RobotSpacePoint(x, y, z, grp_hdg)
        joints = arm.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)
        self.assertAlmostEqual(joints.z, 0.2)
        self.assertAlmostEqual(joints.theta3, 0)

    def test_fwdkin_origin(self):
        """
        Checks that forward kinematics works with origin not at 0
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, origin=Vector3D(1,1,1))

        th1 = pi / 2
        th2 = pi / 2
        z = 0.1
        th3 = pi / 2
        joints = JointSpacePoint(th1, th2, z, th3)
        tool = arm.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, -l2 + 1)
        self.assertAlmostEqual(tool.y, l1 + 1)
        self.assertAlmostEqual(tool.z, 0.1 + 1)
        self.assertAlmostEqual(tool.gripper_hdg, -pi)

        th1 = -pi / 2
        th2 = pi / 2
        z = 0.2
        th3 = pi / 2
        joints = JointSpacePoint(th1, th2, z, th3)
        tool = arm.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l2 + 1)
        self.assertAlmostEqual(tool.y, -l1 + 1)
        self.assertAlmostEqual(tool.z, 0.2 + 1)
        self.assertAlmostEqual(tool.gripper_hdg, 0.0)

    def test_invkin_origin(self):
        """
        Checks that inverse kinematics works with origin not at 0
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, origin=Vector3D(1,1,1))
        x = 1 + l2
        y = 1 - l1
        z = 1 + 0.1
        grp_hdg = 0
        tool = RobotSpacePoint(x, y, z, grp_hdg)
        joints = arm.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)
        self.assertAlmostEqual(joints.z, 0.1)
        self.assertAlmostEqual(joints.theta3, pi / 2)

        x = 1
        y = 1 + l1 + l2
        z = 1 + 0.2
        grp_hdg = 0
        tool = RobotSpacePoint(x, y, z, grp_hdg)
        joints = arm.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0)
        self.assertAlmostEqual(joints.z, 0.2)
        self.assertAlmostEqual(joints.theta3, 0)

    def test_target_unreachable(self):
        """
        Checks that an out of range target is detected as unreachable
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        tool = RobotSpacePoint(1+l1+l2, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = arm.inverse_kinematics(tool)

        tool = RobotSpacePoint(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = arm.inverse_kinematics(tool)

    def test_jacobian(self):
        """
        Checks that jacobian matrix is correct
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        jacobian = arm.compute_jacobian()
        self.assertAlmostEqual(jacobian[0,0], 0)
        self.assertAlmostEqual(jacobian[0,1], 0)
        self.assertAlmostEqual(jacobian[1,0], l1 + l2)
        self.assertAlmostEqual(jacobian[1,1], l2)
        self.assertAlmostEqual(jacobian[2,0], 0)
        self.assertAlmostEqual(jacobian[2,1], 0)
        self.assertAlmostEqual(jacobian[2,2], 1)
        self.assertAlmostEqual(jacobian[2,3], 0)
        self.assertAlmostEqual(jacobian[3,0], -1)
        self.assertAlmostEqual(jacobian[3,1], -1)
        self.assertAlmostEqual(jacobian[3,2], 0)
        self.assertAlmostEqual(jacobian[3,3], -1)

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = arm.forward_kinematics(joints)
        jacobian = arm.compute_jacobian()
        self.assertAlmostEqual(jacobian[0,0], - l2)
        self.assertAlmostEqual(jacobian[0,1], - l2)
        self.assertAlmostEqual(jacobian[1,0], l1)
        self.assertAlmostEqual(jacobian[1,1], 0)
        self.assertAlmostEqual(jacobian[2,0], 0)
        self.assertAlmostEqual(jacobian[2,1], 0)
        self.assertAlmostEqual(jacobian[2,2], 1)
        self.assertAlmostEqual(jacobian[2,3], 0)
        self.assertAlmostEqual(jacobian[3,0], -1)
        self.assertAlmostEqual(jacobian[3,1], -1)
        self.assertAlmostEqual(jacobian[3,2], 0)
        self.assertAlmostEqual(jacobian[3,3], -1)

    def test_tool_vel(self):
        """
        Checks that tool velocity is correctly estimated
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        joints_vel = np.matrix([[0], [0], [0], [0]])
        tool_vel = arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel[0], 0)
        self.assertAlmostEqual(tool_vel[1], 0)
        self.assertAlmostEqual(tool_vel[2], 0)
        self.assertAlmostEqual(tool_vel[3], 0)

        joints_vel = np.matrix([[1], [1], [1], [1]])
        tool_vel = arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel[0], 0)
        self.assertAlmostEqual(tool_vel[1], (l1 + l2) * joints_vel[0] \
                                            + l2 * joints_vel[1])
        self.assertAlmostEqual(tool_vel[2], 1)
        self.assertAlmostEqual(tool_vel[3], - (joints_vel[0] \
                                               + joints_vel[1] \
                                               + joints_vel[2]))

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = arm.forward_kinematics(joints)
        joints_vel = np.matrix([[1], [1], [1], [1]])
        tool_vel = arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel[0], - l2 * (joints_vel[0] + joints_vel[1]))
        self.assertAlmostEqual(tool_vel[1], l1 * joints_vel[0])
        self.assertAlmostEqual(tool_vel[2], 1)
        self.assertAlmostEqual(tool_vel[3], - (joints_vel[0] \
                                               + joints_vel[1] \
                                               + joints_vel[2]))

    def test_joints_vel(self):
        """
        Checks that joints velocity is correctly estimated
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2)

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = arm.forward_kinematics(joints)
        tool_vel = np.matrix([[0], [0], [0], [0]])
        joints_vel = arm.get_joints_vel(tool_vel)
        self.assertAlmostEqual(joints_vel[0], 0)
        self.assertAlmostEqual(joints_vel[1], 0)
        self.assertAlmostEqual(joints_vel[2], 0)
        self.assertAlmostEqual(joints_vel[3], 0)

        joints = JointSpacePoint(0, 0, 0, 0)
        tool = arm.forward_kinematics(joints)
        tool_vel = np.matrix([[1], [1], [1], [1]])
        with self.assertRaises(ValueError):
            joints_vel = arm.get_joints_vel(tool_vel)

        joints = JointSpacePoint(0, pi, 0, 0)
        tool = arm.forward_kinematics(joints)
        tool_vel = np.matrix([[1], [1], [1], [1]])
        with self.assertRaises(ValueError):
            joints_vel = arm.get_joints_vel(tool_vel)
