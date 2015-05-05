from invkin.Datatypes import *
from invkin import Scara
from math import pi, sqrt, cos, sin
import unittest

l1 = 1.0
l2 = 0.5

class ScaraTestCase(unittest.TestCase):
    def test_fwdkin_second_link(self):
        """
        Checks that forward kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = scara.forward_kinematics()
        self.assertAlmostEqual(tool.x, l1 + l2)
        self.assertAlmostEqual(tool.y, 0.0)

        joints = JointSpacePoint(0, -pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, l1 + l2 * cos(joints.theta2))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

        joints = JointSpacePoint(0, pi/4, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, l1 + l2 * cos(joints.theta2))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

    def test_fwdkin_both_links(self):
        """
        Checks that forward kinematics works when both links are moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, -l2)
        self.assertAlmostEqual(tool.y, l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, l2)
        self.assertAlmostEqual(tool.y, -l1)


    def test_invkin_second_link(self):
        """
        Checks that inverse kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(l1+l2, 0, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, 0.0)
        self.assertAlmostEqual(joints.theta2, 0.0)

        tool = RobotSpacePoint(l1, l2, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, 0.0)
        self.assertAlmostEqual(joints.theta2, pi / 2)

    def test_invkin_both_links(self):
        """
        Checks that inverse kinematics works when both links move
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(l2, -l1, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(0, l1+l2, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_fwdkin_flip(self):
        """
        Checks that forward kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, flip_x=-1)

        joints = JointSpacePoint(0, pi/4, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, -(l1 + l2 * cos(joints.theta2)))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, l2)
        self.assertAlmostEqual(tool.y, l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, -l2)
        self.assertAlmostEqual(tool.y, -l1)

    def test_invkin_flip(self):
        """
        Checks that inverse kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, flip_x=-1)

        tool = RobotSpacePoint(-l2, -l1, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(0, l1+l2, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_fwdkin_origin(self):
        """
        Checks that forward kinematics works with origin not at 0
        """
        scara = Scara.Scara(l1=l1, l2=l2, origin=Vector2D(1,1))

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, 1 - l2)
        self.assertAlmostEqual(tool.y, 1 + l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.update_joints(joints)
        self.assertAlmostEqual(tool.x, 1 + l2)
        self.assertAlmostEqual(tool.y, 1 - l1)


    def test_invkin_origin(self):
        """
        Checks that inverse kinematics works with origin not at 0
        """
        scara = Scara.Scara(l1=l1, l2=l2, origin=Vector2D(1,1))

        tool = RobotSpacePoint(1+l2, 1-l1, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(1, 1+l1+l2, 0, 0)
        joints = scara.update_tool(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_target_unreachable(self):
        """
        Checks that an out of range target is detected as unreachable
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(1+l1+l2, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = scara.update_tool(tool)
