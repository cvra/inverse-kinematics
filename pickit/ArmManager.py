from __future__ import division
from pickit.Datatypes import *
from pickit.Scara import Scara
from pickit.DebraArm import DebraArm

class ArmManager(object):
    "Wrapper for arm classes for easier use"

    def __init__(self, arm, workspace, time_resolution):
        if arm.__class__.__name__ == 'DebraArm':
            self.arm = arm
        else:
            raise ValueError('Unhandled arm manipulator class')

        if self.workspace_within_constraints(workspace):
            self.workspace = workspace
        else:
            raise ValueError('Workspace out of constraints')

        self.dt = time_resolution

    def workspace_within_constraints(self, workspace):
        """
        Check that workspace is withing constraints of the arm
        """
        if workspace.x_min < self.arm.x_axis.constraints.pos_min \
            or workspace.x_max > self.arm.x_axis.constraints.pos_max \
            or workspace.y_min < self.arm.y_axis.constraints.pos_min \
            or workspace.y_max > self.arm.y_axis.constraints.pos_max \
            or workspace.z_min < self.arm.z_axis.constraints.pos_min \
            or workspace.z_max > self.arm.z_axis.constraints.pos_max:
            return 0
        else:
            return 1
