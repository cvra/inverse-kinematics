from __future__ import division
from pickit.Datatypes import *
from pickit import Joint, DebraArm, ArmManager
import numpy as np
import unittest

l1 = 1.0
l2 = 1.0
DELTA_T = 0.01

class ArmManagerInitTestCase(unittest.TestCase):
    def test_init(self):
        arm = DebraArm.DebraArm(l1=l1, l2=l2)
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        ws_front = ws
        ws_side = ws
        ws_back = ws

        arm_mng = ArmManager.ArmManager(arm, ws_front, ws_side, ws_back, DELTA_T)

        self.assertAlmostEqual(arm_mng.arm, arm)
        self.assertAlmostEqual(arm_mng.workspace, ws)
        self.assertAlmostEqual(arm_mng.dt, DELTA_T)

    def test_check_arm_class(self):
        arm = Joint.Joint('wrong')
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        ws_front = ws
        ws_side = ws
        ws_back = ws

        with self.assertRaises(ValueError):
            arm_mng = ArmManager.ArmManager(arm, ws_front, ws_side, ws_back, DELTA_T)

class ArmManagerWorkspaceWithinConstraintsTestCase(unittest.TestCase):
    def setUp(self):
        arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        ws_front = self.ws
        ws_side = self.ws
        ws_back = self.ws
        self.arm_mng = ArmManager.ArmManager(arm, ws_front, ws_side, ws_back, DELTA_T)

    def test_is_ok(self):
        ws = self.ws
        self.assertTrue(self.arm_mng.workspace_within_constraints(ws))

    def test_x_min_out(self):
        ws = self.ws._replace(x_min = -3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_x_max_out(self):
        ws = self.ws._replace(x_max = 3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_y_min_out(self):
        ws = self.ws._replace(y_min = -3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_y_max_out(self):
        ws = self.ws._replace(y_max = 3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_z_min_out(self):
        ws = self.ws._replace(z_min= -3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_z_max_out(self):
        ws = self.ws._replace(z_max= 3.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

class ArmManagerClipWorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        ws_front = self.ws
        ws_side = self.ws
        ws_back = self.ws
        self.arm_mng = ArmManager.ArmManager(arm, ws_front, ws_side, ws_back, DELTA_T)

    def test_is_ok(self):
        ws = self.ws
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_x_min_out(self):
        ws = self.ws._replace(x_min = -3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertNotEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_x_max_out(self):
        ws = self.ws._replace(x_max = 3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertNotEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_y_min_out(self):
        ws = self.ws._replace(y_min = -3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertNotEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_y_max_out(self):
        ws = self.ws._replace(y_max = 3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertNotEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_z_min_out(self):
        ws = self.ws._replace(z_min = -3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertNotEqual(new_ws.z_min, ws.z_min)
        self.assertAlmostEqual(new_ws.z_max, ws.z_max)

    def test_z_max_out(self):
        ws = self.ws._replace(z_max = 3.0)
        new_ws = self.arm_mng.clip_workspace_to_constraints(ws)
        self.assertAlmostEqual(new_ws.x_min, ws.x_min)
        self.assertAlmostEqual(new_ws.x_max, ws.x_max)
        self.assertAlmostEqual(new_ws.y_min, ws.y_min)
        self.assertAlmostEqual(new_ws.y_max, ws.y_max)
        self.assertAlmostEqual(new_ws.z_min, ws.z_min)
        self.assertNotEqual(new_ws.z_max, ws.z_max)

class ArmManagerWorkspaceContainingPosTestCase(unittest.TestCase):
    def setUp(self):
        self.arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.ws_front = Workspace(-1.0,1.0, 0.2,2.0, 0.0,0.2, 1)
        self.ws_side = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        self.ws_back = Workspace(-1.0,1.0, -2.0,-0.2, 0.0,0.2, -1)
        self.arm_mng = \
            ArmManager.ArmManager(self.arm, self.ws_front, self.ws_side, self.ws_back, DELTA_T)

    def test_pos_in_side(self):
        pos = RobotSpacePoint(0.5, 0.0, 0.1, 0.0)
        est_ws = self.arm_mng.workspace_containing_position(pos)
        self.assertAlmostEqual(est_ws.x_min, self.ws_side.x_min)
        self.assertAlmostEqual(est_ws.x_max, self.ws_side.x_max)
        self.assertAlmostEqual(est_ws.y_min, self.ws_side.y_min)
        self.assertAlmostEqual(est_ws.y_max, self.ws_side.y_max)
        self.assertAlmostEqual(est_ws.z_min, self.ws_side.z_min)
        self.assertAlmostEqual(est_ws.z_max, self.ws_side.z_max)

    def test_pos_in_front(self):
        pos = RobotSpacePoint(0.0, 1.0, 0.1, 0.0)
        est_ws = self.arm_mng.workspace_containing_position(pos)
        self.assertAlmostEqual(est_ws.x_min, self.ws_front.x_min)
        self.assertAlmostEqual(est_ws.x_max, self.ws_front.x_max)
        self.assertAlmostEqual(est_ws.y_min, self.ws_front.y_min)
        self.assertAlmostEqual(est_ws.y_max, self.ws_front.y_max)
        self.assertAlmostEqual(est_ws.z_min, self.ws_front.z_min)
        self.assertAlmostEqual(est_ws.z_max, self.ws_front.z_max)

    def test_pos_in_front(self):
        pos = RobotSpacePoint(0.0, -1.0, 0.1, 0.0)
        est_ws = self.arm_mng.workspace_containing_position(pos)
        self.assertAlmostEqual(est_ws.x_min, self.ws_back.x_min)
        self.assertAlmostEqual(est_ws.x_max, self.ws_back.x_max)
        self.assertAlmostEqual(est_ws.y_min, self.ws_back.y_min)
        self.assertAlmostEqual(est_ws.y_max, self.ws_back.y_max)
        self.assertAlmostEqual(est_ws.z_min, self.ws_back.z_min)
        self.assertAlmostEqual(est_ws.z_max, self.ws_back.z_max)

class ArmManagerGoToPosTestCase(unittest.TestCase):
    def setUp(self):
        self.arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2, 1)
        ws_front = self.ws
        ws_side = self.ws
        ws_back = self.ws
        self.arm_mng = ArmManager.ArmManager(self.arm, ws_front, ws_side, ws_back, DELTA_T)

        self.start_pos = RobotSpacePoint(1.0, -0.5, 0.0, 0.0)
        self.start_vel = RobotSpacePoint(0.0, 0.0, 0.0, 0.0)
        self.target_pos = RobotSpacePoint(1.0, 0.5, 0.0, 0.0)
        self.target_vel = RobotSpacePoint(0.0, 0.0, 0.0, 0.0)

    def test_goto_path_robot_space(self):
        q1, q2, q3, q4 = self.arm_mng.goto_position(self.start_pos,
                                                    self.start_vel,
                                                    self.target_pos,
                                                    self.target_vel,
                                                    'line')
        t1, t2, t3, t4 = self.arm.get_path_xyz(self.start_pos,
                                               self.start_vel,
                                               self.target_pos,
                                               self.target_vel,
                                               DELTA_T,
                                               'joint')

        self.assertAlmostEqual(t1, q1)
        self.assertAlmostEqual(t2, q2)
        self.assertAlmostEqual(t3, q3)
        self.assertAlmostEqual(t4, q4)

    def test_goto_path_joint_space(self):
        q1, q2, q3, q4 = self.arm_mng.goto_position(self.start_pos,
                                                    self.start_vel,
                                                    self.target_pos,
                                                    self.target_vel,
                                                    'curve')
        t1, t2, t3, t4 = self.arm.get_path(self.start_pos,
                                           self.start_vel,
                                           self.target_pos,
                                           self.target_vel,
                                           DELTA_T)

        self.assertAlmostEqual(t1, q1)
        self.assertAlmostEqual(t2, q2)
        self.assertAlmostEqual(t3, q3)
        self.assertAlmostEqual(t4, q4)
