from __future__ import division
from pickit.Datatypes import *
from pickit import DebraArm
from math import pi, sqrt, cos, sin
import numpy as np
import random, time
import unittest

l1 = 1.0
l2 = 0.5
DELTA_T = 0.01

class DebraArmForwardKinematicsTestCase(unittest.TestCase):
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

class DebraArmInverseKinematicsTestCase(unittest.TestCase):
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

class DebraArmJacobianTestCase(unittest.TestCase):
    def setUp(self):
        self.arm = DebraArm.DebraArm(l1=l1, l2=l2)

    def test_jacobian(self):
        """
        Checks that jacobian matrix is correct
        """
        jacobian = self.arm.compute_jacobian()
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
        tool = self.arm.forward_kinematics(joints)
        jacobian = self.arm.compute_jacobian()
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
        joints_vel = JointSpacePoint(0, 0, 0, 0)
        jacobian = self.arm.compute_jacobian()
        tool_vel = self.arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, 0)
        self.assertAlmostEqual(tool_vel.y, 0)
        self.assertAlmostEqual(tool_vel.z, 0)
        self.assertAlmostEqual(tool_vel.gripper_hdg, 0)

        joints_vel = JointSpacePoint(1, 1, 1, 1)
        jacobian = self.arm.compute_jacobian()
        tool_vel = self.arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, 0)
        self.assertAlmostEqual(tool_vel.y, (l1 + l2) * joints_vel.theta1 \
                                            + l2 * joints_vel.theta2)
        self.assertAlmostEqual(tool_vel.z, 1)
        self.assertAlmostEqual(tool_vel.gripper_hdg, - (joints_vel.theta1 \
                                                       + joints_vel.theta2 \
                                                       + joints_vel.theta3))

        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = self.arm.forward_kinematics(joints)
        joints_vel = JointSpacePoint(1, 1, 1, 1)
        jacobian = self.arm.compute_jacobian()
        tool_vel = self.arm.get_tool_vel(joints_vel)
        self.assertAlmostEqual(tool_vel.x, - l2 * (joints_vel.theta1 + joints_vel.theta2))
        self.assertAlmostEqual(tool_vel.y, l1 * joints_vel.theta1)
        self.assertAlmostEqual(tool_vel.z, 1)
        self.assertAlmostEqual(tool_vel.gripper_hdg, - (joints_vel.theta1 \
                                                       + joints_vel.theta2 \
                                                       + joints_vel.theta3))

    def test_joints_vel(self):
        """
        Checks that joints velocity is correctly estimated
        """
        joints = JointSpacePoint(0, pi/2, 0, 0)
        tool = self.arm.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(0, 0, 0, 0)
        jacobian_inv = self.arm.compute_jacobian_inv()
        joints_vel = self.arm.get_joints_vel(tool_vel)
        self.assertAlmostEqual(joints_vel.theta1, 0)
        self.assertAlmostEqual(joints_vel.theta2, 0)
        self.assertAlmostEqual(joints_vel.theta3, 0)
        self.assertAlmostEqual(joints_vel.z, 0)

        joints = JointSpacePoint(0, 0, 0, 0)
        tool = self.arm.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(1, 1, 1, 1)
        with self.assertRaises(ValueError):
            jacobian_inv = self.arm.compute_jacobian_inv()
            joints_vel = self.arm.get_joints_vel(tool_vel)

        joints = JointSpacePoint(0, pi, 0, 0)
        tool = self.arm.forward_kinematics(joints)
        tool_vel = RobotSpacePoint(1, 1, 1, 1)
        with self.assertRaises(ValueError):
            jacobian_inv = self.arm.compute_jacobian_inv()
            joints_vel = self.arm.get_joints_vel(tool_vel)

class DebraArmTrajectoryGenerationTestCase(unittest.TestCase):
    def setUp(self):
        self.arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.arm.inverse_kinematics(RobotSpacePoint(0.99*(l1+l2), 0, 0, 0))

    def test_target_unreachable(self):
        """
        Checks that an out of range target is detected as unreachable
        """
        tool = RobotSpacePoint(1+l1+l2, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = self.arm.inverse_kinematics(tool)

        tool = RobotSpacePoint(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            joints = self.arm.inverse_kinematics(tool)

    def test_path_xyz(self):
        """
        Checks that path generation in xyz outputs a trajectory that satisfies
        initial and final states
        """
        tool = self.arm.get_tool()

        start_vel = RobotSpacePoint(0,0,0,0)
        target_vel = RobotSpacePoint(0,0,0,0)

        random.seed(42)
        timing = []
        print('\n')

        # Generate random trajectories
        for i in range(100):
            tool_prev = tool

            th_prev = np.arctan2(tool_prev.y, tool_prev.x)
            r = random.uniform(abs(l1-l2), abs(l1+l2))
            dth = random.uniform(-pi / 3, pi / 3)

            x = r * np.cos(th_prev + dth)
            y = r * np.sin(th_prev + dth)
            z = random.uniform(self.arm.z_axis.constraints.pos_min,
                               self.arm.z_axis.constraints.pos_max)
            grp = random.uniform(self.arm.gripper_axis.constraints.pos_min,
                                 self.arm.gripper_axis.constraints.pos_max)

            tool = RobotSpacePoint(x, y, z, grp)
            # print('#', i, 'moving from:', (tool_prev.x, tool_prev.y), 'to:', (x, y))

            starting_time = time.time()
            px, py, pz, pgrp = self.arm.get_path_xyz(tool_prev,
                                                     RobotSpacePoint(0,0,0,0),
                                                     tool,
                                                     RobotSpacePoint(0,0,0,0),
                                                     DELTA_T,
                                                     'robot')
            ending_time = time.time()
            timing.append(ending_time - starting_time)

            # Check start
            self.assertAlmostEqual(px[0][1], tool_prev.x, places=3)
            self.assertAlmostEqual(py[0][1], tool_prev.y, places=3)
            self.assertAlmostEqual(pz[0][1], tool_prev.z, places=3)
            self.assertAlmostEqual(pgrp[0][1], tool_prev.gripper_hdg, places=3)
            # Check arrival
            self.assertAlmostEqual(px[-1][1], tool.x, places=3)
            self.assertAlmostEqual(py[-1][1], tool.y, places=3)
            self.assertAlmostEqual(pz[-1][1], tool.z, places=3)
            self.assertAlmostEqual(pgrp[-1][1], tool.gripper_hdg, places=3)

        print('Computed ', len(timing), ' trajectories in robot space successfully')
        print('Average computation time:', int(np.mean(timing) * 1e6), 'us',
              '+/-', int(np.std(timing) * 1e6), 'us')

if __name__ == '__main__':
    unittest.main()
