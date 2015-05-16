from __future__ import division
from pickit.Datatypes import *
from pickit.Joint import Joint
from pickit.Scara import Scara
import numpy as np

class DebraArm(Scara):
    "Kinematics and trajectory generation of an arm on Debra (3dof + hand)"

    def __init__(self, l1=1.0, l2=1.0,
                 theta1_constraints=JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
                 theta2_constraints=JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
                 theta3_constraints=JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
                 z_constraints=JointMinMaxConstraint(0,1, -1,1, -1,1),
                 q0=JointSpacePoint(0,0,0,0),
                 origin=Vector3D(0,0,0),
                 flip_x=FLIP_RIGHT_HAND):
        """
        Input:
        l1 - length of first link
        l2 - lenght of second link
        q0 - initial positions of joints
        origin - position of the base of the arm in carthesian space
        flip_x - vertical flip (positive for right hand, negative for left hand)
        """
        self.l1 = l1
        self.l2 = l2
        self.lsq = l1 ** 2 + l2 ** 2
        self.joints = q0
        self.origin = origin

        self.theta1_joint = Joint('theta1', theta1_constraints)
        self.theta2_joint = Joint('theta2', theta2_constraints)
        self.theta3_joint = Joint('theta3', theta3_constraints)
        self.z_joint = Joint('z', z_constraints)

        self.x_axis = Joint('x', JointMinMaxConstraint(-(l1+l2),l1+l2, -1,1, -1,1))
        self.y_axis = Joint('y', JointMinMaxConstraint(-(l1+l2),l1+l2, -1,1, -1,1))
        self.z_axis = Joint('z', z_constraints)
        self.gripper_axis = Joint('gripper', theta3_constraints)

        vel_con = np.sqrt((l1 * theta1_constraints.vel_max) ** 2
                       + (l2 * theta2_constraints.vel_max) ** 2)

        acc_con = np.sqrt((l1 * theta1_constraints.acc_max) ** 2
                       + (l2 * theta2_constraints.acc_max) ** 2)

        self.path_constraints = JointMinMaxConstraint(-10*(l1+l2), 10*(l1+l2),
                                                      -vel_con, vel_con,
                                                      -acc_con, acc_con)

        self.jacobian = np.matrix([[ 1.0,  0.0, 0.0,  0.0], \
                                   [ 0.0,  1.0, 0.0,  0.0], \
                                   [ 0.0,  0.0, 1.0,  0.0], \
                                   [-1.0, -1.0, 0.0, -1.0]])
        self.jacobian_inv = np.matrix([[1.0, 0.0, 0.0,  0.0], \
                                       [0.0, 1.0, 0.0,  0.0], \
                                       [0.0, 0.0, 1.0,  0.0], \
                                       [0.0, 0.0, 0.0, -1.0]])

        if flip_x > 0:
            self.flip_x = FLIP_RIGHT_HAND
        else:
            self.flip_x = FLIP_LEFT_HAND

        self.flip_elbow = ELBOW_BACK

        self.tool = self.get_tool()

    def forward_kinematics(self, new_joints):
        """
        Update the joint values through computation of forward kinematics
        """
        self.joints = new_joints
        self.tool = self.get_tool()

        return self.tool

    def inverse_kinematics(self, new_tool):
        """
        Update the tool position through computation of inverse kinematics
        """
        norm = (new_tool.x - self.origin.x) ** 2 + (new_tool.y - self.origin.y) ** 2
        if(norm > (self.l1 + self.l2) ** 2 or norm < (self.l1 - self.l2) ** 2):
            "Target unreachable"
            self.tool = self.get_tool()
            raise ValueError('Target unreachable')

        self.tool = new_tool
        self.joints = self.get_joints()

        return self.joints

    def get_tool(self):
        """
        Computes tool position knowing joint positions
        """
        tool = super(DebraArm, self).get_tool()

        grp_hdg = (pi / 2) - (self.joints.theta1 \
                              + self.joints.theta2 \
                              + self.joints.theta3)
        tool = tool._replace(z=(self.joints.z + self.origin.z), gripper_hdg=grp_hdg)

        return tool

    def get_joints(self):
        """
        Computes joint positions knowing tool position
        """
        x = self.tool.x - self.origin.x
        y = self.tool.y - self.origin.y
        z = self.tool.z - self.origin.z
        gripper_hdg = self.tool.gripper_hdg

        joints = super(DebraArm, self).get_joints()

        theta3 = pi / 2 - (gripper_hdg + joints.theta1 + joints.theta2)
        theta3 = (theta3 + pi) % (2 * pi) - pi # Stay between -pi and pi
        joints = joints._replace(z=z, theta3=theta3)

        return joints

    def get_detailed_pos(self, l3):
        """
        Returns origin, p1, p2, p3, z
        """
        p0, p1, p2 = super(DebraArm, self).get_detailed_pos()

        x3 = self.tool.x + self.flip_x * l3 * np.cos(self.joints.theta1 \
                                                  + self.joints.theta2 \
                                                  + self.joints.theta3)
        y3 = self.tool.y + l3 * np.sin(self.joints.theta1 \
                                    + self.joints.theta2 \
                                    + self.joints.theta3)

        return self.origin, p1, p2, Vector2D(x3, y3), self.tool.z

    def compute_jacobian(self):
        """
        Returns jacobian matrix at current state
        """
        sin_th1_th2 = np.sin(self.joints.theta1 + self.joints.theta2)
        cos_th1_th2 = np.cos(self.joints.theta1 + self.joints.theta2)

        self.jacobian[0,0] = - self.l1 * np.sin(self.joints.theta1) \
                             - self.l2 * sin_th1_th2
        self.jacobian[0,1] = - self.l2 * sin_th1_th2

        self.jacobian[1,0] = self.l1 * np.cos(self.joints.theta1) \
                             + self.l2 * cos_th1_th2
        self.jacobian[1,1] = self.l2 * cos_th1_th2

        return self.jacobian

    def compute_jacobian_inv(self):
        """
        Returns the inverse of the jacobian matrix at current state
        """
        if abs(self.joints.theta2) < EPSILON \
           or abs(self.joints.theta2 - pi) < EPSILON:
            # raise ValueError('Singularity')
            return self.jacobian_inv

        sin_th1_th2 = np.sin(self.joints.theta1 + self.joints.theta2)
        cos_th1_th2 = np.cos(self.joints.theta1 + self.joints.theta2)

        a = - self.l1 * np.sin(self.joints.theta1) - self.l2 * sin_th1_th2
        b = - self.l2 * sin_th1_th2
        c = self.l1 * np.cos(self.joints.theta1) + self.l2 * cos_th1_th2
        d = self.l2 * cos_th1_th2

        det_inv = 1 / (self.l1 * self.l2 * np.sin(self.joints.theta2))

        self.jacobian_inv[0,0] = d * det_inv
        self.jacobian_inv[0,1] = -b * det_inv
        self.jacobian_inv[1,0] = -c * det_inv
        self.jacobian_inv[1,1] = a * det_inv
        self.jacobian_inv[3,0] = - (self.jacobian_inv[0,0] + self.jacobian_inv[1,0])
        self.jacobian_inv[3,1] = - (self.jacobian_inv[0,1] + self.jacobian_inv[1,1])

        return self.jacobian_inv

    def get_tool_vel(self, joints_vel):
        """
        Computes current tool velocity using jacobian
        """
        joints_vel = np.matrix([[joints_vel.theta1], \
                                [joints_vel.theta2], \
                                [joints_vel.z], \
                                [joints_vel.theta3]])
        tool_vel = self.jacobian * joints_vel

        return RobotSpacePoint(float(tool_vel[0]), \
                               float(tool_vel[1]), \
                               float(tool_vel[2]), \
                               float(tool_vel[3]))

    def get_joints_vel(self, tool_vel):
        """
        Computes current tool velocity using jacobian
        """
        tool_vel = np.matrix([[tool_vel.x], \
                              [tool_vel.y], \
                              [tool_vel.z], \
                              [tool_vel.gripper_hdg]])

        if np.linalg.norm(tool_vel) < EPSILON:
            return JointSpacePoint(0, 0, 0, 0)

        joints_vel = self.jacobian_inv * tool_vel

        return JointSpacePoint(float(joints_vel[0]), \
                               float(joints_vel[1]), \
                               float(joints_vel[2]), \
                               float(joints_vel[3]))

    def get_path(self, start_pos, start_vel, target_pos, target_vel, delta_t):
        """
        Generates a time optimal trajectory for the whole arm
        Input:
        start_pos - start position in tool space
        start_vel - start velocity in tool space
        target_pos - target position in tool space
        target_vel - target velocity in tool space
        """
        # Determine current (start) state and final (target) state
        start_joints_pos = self.inverse_kinematics(start_pos)
        self.compute_jacobian_inv()
        start_joints_vel = self.get_joints_vel(start_vel)

        target_joints_pos = self.inverse_kinematics(target_pos)
        self.compute_jacobian_inv()
        target_joints_vel = self.get_joints_vel(target_vel)

        # Get synchronisation time
        tf_sync = self.synchronisation_time(start_joints_pos,
                                            start_joints_vel,
                                            target_joints_pos,
                                            target_joints_vel)

        # Get trajectories for each joint
        traj_theta1 = self.theta1_joint.get_path(start_joints_pos.theta1,
                                                 start_joints_vel.theta1,
                                                 target_joints_pos.theta1,
                                                 target_joints_vel.theta1,
                                                 tf_sync,
                                                 delta_t)

        traj_theta2 = self.theta2_joint.get_path(start_joints_pos.theta2,
                                                 start_joints_vel.theta2,
                                                 target_joints_pos.theta2,
                                                 target_joints_vel.theta2,
                                                 tf_sync,
                                                 delta_t)

        traj_z = self.z_joint.get_path(start_joints_pos.z,
                                       start_joints_vel.z,
                                       target_joints_pos.z,
                                       target_joints_vel.z,
                                       tf_sync,
                                       delta_t)

        traj_theta3 = self.theta3_joint.get_path(start_joints_pos.theta3,
                                                 start_joints_vel.theta3,
                                                 target_joints_pos.theta3,
                                                 target_joints_vel.theta3,
                                                 tf_sync,
                                                 delta_t)

        return traj_theta1, traj_theta2, traj_z, traj_theta3

    def synchronisation_time(self, start_pos, start_vel, target_pos, target_vel):
        """
        Return largest time to destination to use slowest joint as synchronisation
        reference
        """
        # Compute time to destination for all joints
        ttd_theta1 = self.theta1_joint.time_to_destination(start_pos.theta1,
                                                           start_vel.theta1,
                                                           target_pos.theta1,
                                                           target_vel.theta1)

        ttd_theta2 = self.theta2_joint.time_to_destination(start_pos.theta2,
                                                           start_vel.theta2,
                                                           target_pos.theta2,
                                                           target_vel.theta2)

        ttd_z = self.z_joint.time_to_destination(start_pos.z,
                                                 start_vel.z,
                                                 target_pos.z,
                                                 target_vel.z)

        ttd_theta3 = self.theta3_joint.time_to_destination(start_pos.theta3,
                                                           start_vel.theta3,
                                                           target_pos.theta3,
                                                           target_vel.theta3)

        # Return the largest one
        return np.amax([ttd_theta1.tf, ttd_theta2.tf, ttd_theta3.tf, ttd_z.tf])

    def get_path_xyz(self, start_pos, start_vel, target_pos, target_vel, delta_t,
                     output='joint'):
        """
        Generates a trajectory for the whole arm in robot space
        Input:
        start_pos  - start position in tool space
        start_vel  - start velocity in tool space
        target_pos - target position in tool space
        target_vel - target velocity in tool space
        delta_t    - time step
        output     - select between tool space and joint space trajectories
        """
        # Project constraints on the path to constraints on the axis
        self.path_to_axis_constraint(start_pos, target_pos)

        # Determine current (start) state and final (target) state
        start_joints_pos = self.inverse_kinematics(start_pos)
        self.compute_jacobian_inv()
        start_joints_vel = self.get_joints_vel(start_vel)

        target_joints_pos = self.inverse_kinematics(target_pos)
        self.compute_jacobian_inv()
        target_joints_vel = self.get_joints_vel(target_vel)

        # Get synchronisation time
        tf_sync = self.sync_time_xyz(start_pos, start_vel, target_pos, target_vel)

        # Get trajectories for each joint
        traj_x = self.x_axis.get_path(start_pos.x,
                                      start_vel.x,
                                      target_pos.x,
                                      target_vel.x,
                                      tf_sync,
                                      delta_t)

        traj_y = self.y_axis.get_path(start_pos.y,
                                      start_vel.y,
                                      target_pos.y,
                                      target_vel.y,
                                      tf_sync,
                                      delta_t)

        traj_z = self.z_axis.get_path(start_pos.z,
                                      start_vel.z,
                                      target_pos.z,
                                      target_vel.z,
                                      tf_sync,
                                      delta_t)

        traj_gripper = self.gripper_axis.get_path(start_pos.gripper_hdg,
                                                  start_vel.gripper_hdg,
                                                  target_pos.gripper_hdg,
                                                  target_vel.gripper_hdg,
                                                  tf_sync,
                                                  delta_t)

        th1, th2, z, th3, px, py, pz, pgrp = \
            self.xyz_to_joint_trajectory(traj_x, traj_y, traj_z, traj_gripper)

        if output == 'robot' or output == 'tool':
            return px, py, pz, pgrp
        elif output == 'all' or output == 'both':
            return th1, th2, z, th3, px, py, pz, pgrp
        else:
            return th1, th2, z, th3

    def sync_time_xyz(self, start_pos, start_vel, target_pos, target_vel):
        """
        Return largest time to destination to use slowest axis as synchronisation
        reference
        """
        # Compute time to destination for all joints
        ttd_x = self.x_axis.time_to_destination(start_pos.x, start_vel.x,
                                                target_pos.x, target_vel.x)

        ttd_y = self.y_axis.time_to_destination(start_pos.y, start_vel.y,
                                                target_pos.y, target_vel.y)

        ttd_z = self.z_axis.time_to_destination(start_pos.z, start_vel.z,
                                                target_pos.z, target_vel.z)

        ttd_gripper = self.gripper_axis.time_to_destination(start_pos.gripper_hdg,
                                                            start_vel.gripper_hdg,
                                                            target_pos.gripper_hdg,
                                                            target_vel.gripper_hdg)

        # Return the largest one
        return np.amax([ttd_x.tf, ttd_y.tf, ttd_z.tf, ttd_gripper.tf])

    def xyz_to_joint_trajectory(self, points_x, points_y, points_z, points_gripper):
        """
        Convert trajectory from robot space to joint space
        """
        traj_x = []
        traj_y = []
        traj_z = []
        traj_gripper = []

        traj_joint_th1 = []
        traj_joint_th2 = []
        traj_joint_th3 = []
        traj_joint_z = []

        for x, y, z, grp in zip(points_x, points_y, points_z, points_gripper):
            pos = RobotSpacePoint(x[1], y[1], z[1], grp[1])
            vel = RobotSpacePoint(x[2], y[2], z[2], grp[2])
            acc = RobotSpacePoint(x[3], y[3], z[3], grp[3])

            joints_pos = self.inverse_kinematics(pos)
            self.compute_jacobian_inv()
            joints_vel = self.get_joints_vel(vel)
            joints_acc = self.get_joints_vel(acc)

            traj_joint_th1.append((x[0], joints_pos[0], joints_vel[0], joints_acc[0]))
            traj_joint_th2.append((x[0], joints_pos[1], joints_vel[1], joints_acc[1]))
            traj_joint_th3.append((x[0], joints_pos[3], joints_vel[3], joints_acc[3]))
            traj_joint_z.append((x[0], joints_pos[2], joints_vel[2], joints_acc[2]))

            # Rebuild original xyz trajectory
            traj_x.append((x[0], x[1], x[2], x[3]))
            traj_y.append((y[0], y[1], y[2], y[3]))
            traj_z.append((z[0], z[1], z[2], z[3]))
            traj_gripper.append((grp[0], grp[1], grp[2], grp[3]))

        return traj_joint_th1, traj_joint_th2, traj_joint_z, traj_joint_th3, \
               traj_x, traj_y, traj_z, traj_gripper

    def path_to_axis_constraint(self, start_pos, target_pos):
        """
        Project path constraints to constraints on the axis
        """
        delta_x = abs(target_pos.x - start_pos.x)
        delta_y = abs(target_pos.y - start_pos.y)

        delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2)

        coef_x = delta_x / delta_s
        coef_y = delta_y / delta_s

        if coef_x > EPSILON:
            self.x_axis.set_constraints(
                self.x_axis.constraints._replace(
                    vel_min=(self.path_constraints.vel_min * coef_x),
                    vel_max=(self.path_constraints.vel_max * coef_x),
                    acc_min=(self.path_constraints.acc_min * coef_x),
                    acc_max=(self.path_constraints.acc_max * coef_x)
                    )
                )

        if coef_y > EPSILON:
            self.y_axis.set_constraints(
                self.y_axis.constraints._replace(
                    vel_min=(self.path_constraints.vel_min * coef_y),
                    vel_max=(self.path_constraints.vel_max * coef_y),
                    acc_min=(self.path_constraints.acc_min * coef_y),
                    acc_max=(self.path_constraints.acc_max * coef_y)
                    )
                )
