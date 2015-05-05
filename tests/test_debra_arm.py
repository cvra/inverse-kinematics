from invkin.Datatypes import *
from invkin import DebraArm
from math import pi, sqrt, cos, sin
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

