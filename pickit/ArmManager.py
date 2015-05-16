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
            self.workspace = self.clip_workspace_to_constraints(workspace)

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

    def clip_workspace_to_constraints(self, workspace):
        """
        Clips the workspace to make it fit within constraints of the arm
        """
        x_min = max(workspace.x_min, self.arm.x_axis.constraints.pos_min)
        x_max = min(workspace.x_max, self.arm.x_axis.constraints.pos_max)
        y_min = max(workspace.y_min, self.arm.y_axis.constraints.pos_min)
        y_max = min(workspace.y_max, self.arm.y_axis.constraints.pos_max)
        z_min = max(workspace.z_min, self.arm.z_axis.constraints.pos_min)
        z_max = min(workspace.z_max, self.arm.z_axis.constraints.pos_max)

        return Workspace(x_min, x_max, y_min, y_max, z_min, z_max)

    def goto_position(self, start_pos, start_vel, target_pos, target_vel,
                      shape='line'):
        """
        Return the trajectory to move from start to target
        """
        if shape == 'line' or shape == 'straight' or shape == 'xyz':
            return self.arm.get_path_xyz(start_pos,
                                         start_vel,
                                         target_pos,
                                         target_vel,
                                         self.dt,
                                         'joint')
        elif shape == 'curve' or shape == 'joint':
            return self.arm.get_path(start_pos,
                                     start_vel,
                                     target_pos,
                                     target_vel,
                                     self.dt)
        else:
            raise ValueError('Unknown shape of trajectory requested')
