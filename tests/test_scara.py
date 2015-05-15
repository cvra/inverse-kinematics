from pickit.Datatypes import *
from pickit import Scara
from math import pi, sqrt, cos, sin
import numpy as np
import unittest

l1 = 1.0
l2 = 0.5

class ScaraTestCase(unittest.TestCase):
    def test_fwdkin_second_link(self):
        """
        Checks that forward kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = scara.get_tool()
        self.assertAlmostEqual(tool.x, l1 + l2)
        self.assertAlmostEqual(tool.y, 0.0)

        joints = JointSpacePoint(0, -pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l1 + l2 * cos(joints.theta2))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

        joints = JointSpacePoint(0, pi/4, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l1 + l2 * cos(joints.theta2))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

    def test_fwdkin_both_links(self):
        """
        Checks that forward kinematics works when both links are moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, -l2)
        self.assertAlmostEqual(tool.y, l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l2)
        self.assertAlmostEqual(tool.y, -l1)


    def test_invkin_second_link(self):
        """
        Checks that inverse kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(l1+l2, 0, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, 0.0)
        self.assertAlmostEqual(joints.theta2, 0.0)

        tool = RobotSpacePoint(l1, l2, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, 0.0)
        self.assertAlmostEqual(joints.theta2, pi / 2)

    def test_invkin_both_links(self):
        """
        Checks that inverse kinematics works when both links move
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(l2, -l1, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(0, l1+l2, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_fwdkin_flip(self):
        """
        Checks that forward kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, flip_x=-1)

        joints = JointSpacePoint(0, pi/4, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, -(l1 + l2 * cos(joints.theta2)))
        self.assertAlmostEqual(tool.y, l2 * sin(joints.theta2))

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, l2)
        self.assertAlmostEqual(tool.y, l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, -l2)
        self.assertAlmostEqual(tool.y, -l1)

    def test_invkin_flip(self):
        """
        Checks that inverse kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, flip_x=-1)

        tool = RobotSpacePoint(-l2, -l1, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(0, l1+l2, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_fwdkin_origin(self):
        """
        Checks that forward kinematics works with origin not at 0
        """
        scara = Scara.Scara(l1=l1, l2=l2, origin=Vector2D(1,1))

        joints = JointSpacePoint(pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, 1 - l2)
        self.assertAlmostEqual(tool.y, 1 + l1)

        joints = JointSpacePoint(-pi/2, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        self.assertAlmostEqual(tool.x, 1 + l2)
        self.assertAlmostEqual(tool.y, 1 - l1)


    def test_invkin_origin(self):
        """
        Checks that inverse kinematics works with origin not at 0
        """
        scara = Scara.Scara(l1=l1, l2=l2, origin=Vector2D(1,1))

        tool = RobotSpacePoint(1+l2, 1-l1, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, -pi / 2)
        self.assertAlmostEqual(joints.theta2, pi / 2)

        tool = RobotSpacePoint(1, 1+l1+l2, 0, 0)
        joints = scara.inverse_kinematics(tool)
        self.assertAlmostEqual(joints.theta1, pi / 2)
        self.assertAlmostEqual(joints.theta2, 0.0)

    def test_target_unreachable(self):
        """
        Checks that an out of range target is detected as unreachable
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        tool = RobotSpacePoint(1+l1+l2, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = scara.inverse_kinematics(tool)

        tool = RobotSpacePoint(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = scara.inverse_kinematics(tool)

    def test_jacobian(self):
        """
        Checks that jacobian matrix is correct
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        jacobian = scara.compute_jacobian()
        self.assertAlmostEqual(jacobian[0,0], 0)
        self.assertAlmostEqual(jacobian[0,1], 0)
        self.assertAlmostEqual(jacobian[1,0], l1 + l2)
        self.assertAlmostEqual(jacobian[1,1], l2)

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        jacobian = scara.compute_jacobian()
        self.assertAlmostEqual(jacobian[0,0], - l2)
        self.assertAlmostEqual(jacobian[0,1], - l2)
        self.assertAlmostEqual(jacobian[1,0], l1)
        self.assertAlmostEqual(jacobian[1,1], 0)

    def test_tool_vel(self):
        """
        Checks that tool velocity is correctly estimated
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        joints_vel = JointSpacePoint(0, 0, 0, 0)
        tool_vel = scara.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, 0)
        self.assertAlmostEqual(tool_vel.y, 0)

        joints_vel = JointSpacePoint(1, 1, 1, 1)
        tool_vel = scara.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, 0)
        self.assertAlmostEqual(tool_vel.y, (l1 + l2) * joints_vel.theta1 \
                                            + l2 * joints_vel.theta2)

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        joints_vel = JointSpacePoint(1, 1, 1, 1)
        tool_vel = scara.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, - l2 * (joints_vel.theta1 + joints_vel.theta2))
        self.assertAlmostEqual(tool_vel.y, l1 * joints_vel.theta1)

    def test_joints_vel(self):
        """
        Checks that joints velocity is correctly estimated
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = scara.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(0, 0, 0, 0)
        joints_vel = scara.get_joints_vel(tool_vel)
        self.assertAlmostEqual(joints_vel.theta1, 0)
        self.assertAlmostEqual(joints_vel.theta2, 0)

        joints = JointSpacePoint(0, 0, 0, 0)
        tool = scara.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(0.1, 0.1, 0.1, 0.1)
        with self.assertRaises(ValueError):
            joints_vel = scara.get_joints_vel(tool_vel)

        joints = JointSpacePoint(0, pi, 0, 0)
        tool = scara.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(0.1, 0.1, 0.1, 0.1)
        with self.assertRaises(ValueError):
            joints_vel = scara.get_joints_vel(tool_vel)

    def test_sync_time(self):
        """
        Check that time to destination is well computed
        """
        scara = Scara.Scara(l1=l1, l2=l2)

        # Final velocity zero
        start_pos = JointSpacePoint(0, 0, 0, 0)
        start_vel = JointSpacePoint(0, 0, 0, 0)
        target_pos = JointSpacePoint(0.5, 0.75, 0, 0)
        target_vel = JointSpacePoint(0, 0, 0, 0)

        tf = scara.synchronisation_time(start_pos, start_vel,
                                        target_pos, target_vel)
        self.assertAlmostEqual(tf, 1.75)

        # Final velocity non zero
        start_pos = JointSpacePoint(0, 0, 0, 0)
        start_vel = JointSpacePoint(0, 0, 0, 0)
        target_pos = JointSpacePoint(0.5, 0.75, 0, 0)
        target_vel = JointSpacePoint(0.5, 0.5, 0, 0)

        tf = scara.synchronisation_time(start_pos, start_vel,
                                        target_pos, target_vel)
        self.assertAlmostEqual(tf, 1.375)

if __name__ == '__main__':
    unittest.main()
