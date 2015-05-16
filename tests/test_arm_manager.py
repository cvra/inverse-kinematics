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
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2)
        arm_mng = ArmManager.ArmManager(arm, ws, DELTA_T)

        self.assertAlmostEqual(arm_mng.arm, arm)
        self.assertAlmostEqual(arm_mng.workspace, ws)
        self.assertAlmostEqual(arm_mng.dt, DELTA_T)

    def test_check_arm_class(self):
        arm = Joint.Joint('wrong')
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2)

        with self.assertRaises(ValueError):
            arm_mng = ArmManager.ArmManager(arm, ws, DELTA_T)

class ArmManagerWorkspaceWithinConstraintsTestCase(unittest.TestCase):
    def setUp(self):
        self.arm = DebraArm.DebraArm(l1=l1, l2=l2)
        self.ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2)
        self.arm_mng = ArmManager.ArmManager(self.arm, self.ws, DELTA_T)

    def test_is_ok(self):
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,0.2)
        self.assertTrue(self.arm_mng.workspace_within_constraints(ws))

    def test_x_min_out(self):
        ws = Workspace(-3.0,2.0, -1.0,1.0, 0.0,0.2)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_x_max_out(self):
        ws = Workspace(0.2,3.0, -1.0,1.0, 0.0,0.2)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_y_min_out(self):
        ws = Workspace(0.2,2.0, -3.0,1.0, 0.0,0.2)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_y_max_out(self):
        ws = Workspace(0.2,2.0, -1.0,3.0, 0.0,0.2)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_z_min_out(self):
        ws = Workspace(0.2,2.0, -1.0,1.0, -1.0,0.2)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))

    def test_z_max_out(self):
        ws = Workspace(0.2,2.0, -1.0,1.0, 0.0,2.0)
        self.assertFalse(self.arm_mng.workspace_within_constraints(ws))
