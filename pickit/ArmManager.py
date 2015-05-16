from __future__ import division
from pickit.Datatypes import *
from pickit.Scara import Scara
from pickit.DebraArm import DebraArm
import numpy as np

def merge_trajectories(traj_a, traj_b):
    """
    Merge two trajectories into one
    """
    traj = []

    for t in traj_a:
        traj.append(t)

    t1, p1, v1, a1 = t

    for t in traj_b:
        time, pos, vel, acc = t
        traj.append((time + t1, pos, vel, acc))

    return traj

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

    def position_within_workspace(self, position, workspace):
        """
        Checks that the position is in the workspace
        """
        if position.x < workspace.x_min \
            or position.x > workspace.x_max \
            or position.y < workspace.y_min \
            or position.y > workspace.y_max \
            or position.z < workspace.z_min \
            or position.z > workspace.z_max:
            return 0
        else:
            return 1

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

    def goto_workspace(self, start_pos, start_vel, target_pos, target_vel,
                       shape, new_workspace, new_elbow_orientation):
        """
        Sets a new workspace for the robot and define the elbow orientation in
        that workspace.
        Outputs a trajectory (adjustment) to go to the new workspace
        """
        # Check that new position is within workspace
        if not self.position_within_workspace(target_pos, new_workspace):
            raise ValueError('Target position not within new workspace boundaries')

        # Register new workspace, if too big, clip it to fit constraints
        if self.workspace_within_constraints(new_workspace):
            self.workspace = new_workspace
        else:
            self.workspace = self.clip_workspace_to_constraints(new_workspace)

        # Compute sequence to move from old workspace to the new position
        # in the new workspace defined
        if np.sign(new_elbow_orientation) == self.arm.flip_elbow:
            return self.goto_position(start_pos, start_vel, target_pos, target_vel, shape)

        # Else, we need to flip the elbow!
        start_joints = self.arm.inverse_kinematics(start_pos)
        inter_joints = start_joints._replace(theta2=0.0)
        inter_pos = self.arm.forward_kinematics(inter_joints)
        inter_vel = RobotSpacePoint(0.0, 0.0, 0.0, 0.0)

        # Go to intermediary point (singularity)
        qa1, qa2, qa3, qa4 = \
            self.goto_position(start_pos, start_vel, inter_pos, inter_vel, 'curve')

        self.arm.flip_elbow *= -1

        # # Go to target
        qb1, qb2, qb3, qb4 = \
            self.goto_position(inter_pos, inter_vel, target_pos, target_vel, shape)

        q1 = merge_trajectories(qa1, qb1)
        q2 = merge_trajectories(qa2, qb2)
        q3 = merge_trajectories(qa3, qb3)
        q4 = merge_trajectories(qa4, qb4)

        # Return trajectory to execute for adjustment
        return q1, q2, q3, q4
