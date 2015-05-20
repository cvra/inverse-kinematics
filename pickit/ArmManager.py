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

    def __init__(self, arm, ws_front, ws_side, ws_back, time_resolution):
        if arm.__class__.__name__ == 'DebraArm':
            self.arm = arm
        else:
            raise ValueError('Unhandled arm manipulator class')

        self.ws_front = self.clip_workspace_to_constraints(ws_front)
        self.ws_side = self.clip_workspace_to_constraints(ws_side)
        self.ws_back = self.clip_workspace_to_constraints(ws_back)

        self.workspace = self.workspace_containing_position(self.arm.get_tool())

        self.tool = arm.get_tool()

        self.dt = time_resolution

    def workspace_containing_position(self, position):
        """
        Returns the workspace containing the position sent
        """
        if self.position_within_workspace(position, self.ws_side):
            return self.ws_side
        elif self.position_within_workspace(position, self.ws_front):
            return self.ws_front
        elif self.position_within_workspace(position, self.ws_back):
            return self.ws_back
        else:
            self.arm.forward_kinematics(JointSpacePoint(0,0,0,0))
            return self.ws_side
            # Force to side workspace

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

        return Workspace(x_min, x_max, y_min, y_max, z_min, z_max,
                         workspace.elbow_orientation)

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

    def workspace_center(self, workspace):
        """
        Return a point at the center of the workspace
        """
        return Vector3D((workspace.x_min + workspace.x_max) / 2,
                        (workspace.y_min + workspace.y_max) / 2,
                        (workspace.z_min + workspace.z_max) / 2)

    def goto(self, start_pos, start_vel, target_pos, target_vel, shape='line'):
        """
        Generic wrapper to move the arm
        """
        new_ws = self.workspace_containing_position(target_pos)

        q1 = []
        q2 = []
        q3 = []
        q4 = []

        print(start_pos, start_vel)

        linear_traj_is_unfeasible = False
        curve_traj_is_unfeasible = False

        try:
            q1, q2, q3, q4 = self.goto_workspace(start_pos, start_vel, target_pos, target_vel, shape, new_ws)
            # Try to go in desired shape
        except ValueError as e:
            print('Can\'t use desired shape, forcing joint space trajectory:', e)
            self.tool = start_pos
            self.arm.inverse_kinematics(self.tool)
            linear_traj_is_unfeasible = True

        if linear_traj_is_unfeasible:
            try:
                q1, q2, q3, q4 =  self.goto_workspace(start_pos, start_vel, target_pos, target_vel, 'curve', new_ws)
                # If can't work it out, force joint space trajectory
            except ValueError as f:
                print('Can\'t go to target:', '... going home')
                self.tool = start_pos
                self.arm.inverse_kinematics(self.tool)
                curve_traj_is_unfeasible = True

        if linear_traj_is_unfeasible and curve_traj_is_unfeasible:
            q1, q2, q3, q4, home_pos = self.go_home(start_pos, start_vel)
            self.tool = home_pos
            self.arm.inverse_kinematics(self.tool)
        else:
            self.tool = target_pos
            self.arm.inverse_kinematics(self.tool)

        return q1, q2, q3, q4

    def go_home(self, start_pos, start_vel):
        """
        Return to safe place: home
        """
        # Define home position as target position
        start_joints_pos = self.arm.inverse_kinematics(start_pos)
        target_joints_pos = JointSpacePoint(-pi/2,
                                            2*pi/3,
                                            start_joints_pos.z,
                                            start_joints_pos.theta3)
        target_pos = self.arm.forward_kinematics(target_joints_pos)
        target_vel = RobotSpacePoint(0, 0, 0, 0)

        new_ws = self.workspace_containing_position(target_pos)

        try:
            q1, q2, q3, q4 = self.goto_workspace(start_pos, start_vel, target_pos, target_vel, 'line', new_ws)
            # Try to go straight
        except ValueError as e:
            self.tool = start_pos
            self.arm.inverse_kinematics(self.tool)
            q1, q2, q3, q4 =  self.goto_workspace(start_pos, start_vel, target_pos, target_vel, 'curve', new_ws)
            # If can't work it out, force joint space trajectory

        return q1, q2, q3, q4, target_pos

    def goto_workspace(self, start_pos, start_vel, target_pos, target_vel,
                       shape, new_workspace):
        """
        Sets a new workspace for the robot and define the elbow orientation in
        that workspace.
        Outputs a trajectory (adjustment) to go to the new workspace
        """
        # Check that new position is within workspace
        if not self.position_within_workspace(target_pos, new_workspace):
            raise ValueError('Target position not within new workspace boundaries, target may be out of defined workspaces')

        self.workspace = new_workspace

        # Compute sequence to move from old workspace to the new position
        # in the new workspace defined
        if np.sign(new_workspace.elbow_orientation) == self.arm.flip_elbow:
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
            self.goto_position(inter_pos, inter_vel, target_pos, target_vel, 'curve')

        q1 = merge_trajectories(qa1, qb1)
        q2 = merge_trajectories(qa2, qb2)
        q3 = merge_trajectories(qa3, qb3)
        q4 = merge_trajectories(qa4, qb4)

        # Return trajectory to execute for adjustment
        return q1, q2, q3, q4

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

    def estimated_time_of_arrival(self, start_pos, start_vel, target_pos, target_vel,
                                  shape='line'):
        """
        Returns time it takes to move from start to target_vel
        """
        if shape == 'curve' or shape == 'joint':
            start_joints_pos = self.arm.inverse_kinematics(start_pos)
            self.arm.compute_jacobian_inv()
            start_joints_vel = self.arm.get_joints_vel(start_vel)

            target_joints_pos = self.arm.inverse_kinematics(target_pos)
            self.arm.compute_jacobian_inv()
            target_joints_vel = self.arm.get_joints_vel(target_vel)

            return self.arm.synchronisation_time(start_joints_pos,
                                                 start_joints_vel,
                                                 target_joints_pos,
                                                 target_joints_vel)
        else:
            return self.arm.sync_time_xyz(start_pos, start_vel, target_pos, target_vel)
