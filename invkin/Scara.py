from invkin.Datatypes import *
from invkin.Constraints import Constraints
from math import sqrt, cos, sin, acos, atan2, pi
import numpy as np

class Scara(object):
    "Kinematics and Inverse kinematics of a Scara (2dof planar arm)"

    def __init__(self, l1=1.0, l2=1.0,
                 theta1_constraints=JointMinMaxConstraint(-pi/2,pi/2, -1,1, -1,1),
                 theta2_constraints=JointMinMaxConstraint(-pi/2,pi/2, -1,1, -1,1),
                 q0=JointSpacePoint(0,0,0,0),
                 origin=Vector2D(0,0),
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

        self.constraints = Constraints()
        self.constraints.add_axis('theta1', theta1_constraints)
        self.constraints.add_axis('theta2', theta2_constraints)

        if flip_x >= 0:
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
            # Target unreachable
            self.tool = self.get_tool()
            raise ValueError('Target unreachable')

        self.tool = new_tool
        self.joints = self.get_joints()

        return self.joints

    def get_tool(self):
        """
        Computes tool position knowing joint positions
        """
        x = self.flip_x * (self.l1 * cos(self.joints.theta1) \
            + self.l2 * cos(self.joints.theta1 + self.joints.theta2))
        y = self.l1 * sin(self.joints.theta1) \
            + self.l2 * sin(self.joints.theta1 + self.joints.theta2)

        x += self.origin.x
        y += self.origin.y

        return RobotSpacePoint(x, y, 0, 0)

    def get_joints(self):
        """
        Computes joint positions knowing tool position
        """
        x = self.tool.x - self.origin.x
        y = self.tool.y - self.origin.y

        if(x == 0 and y == 0):
            return JointSpacePoint(self.joints.theta1, pi, 0, 0)

        l = x ** 2 + y ** 2
        lsq = self.lsq

        cos_gamma = (l + self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * sqrt(l))

        # Numerical errors can make abs(cos_gamma) > 1
        if(cos_gamma > 1 - EPSILON or cos_gamma < -1 + EPSILON):
            gamma = 0.0
        else:
            gamma = self.flip_elbow * acos(cos_gamma)

        theta1 = atan2(y, self.flip_x * x) - gamma
        theta2 = self.flip_elbow * \
                    atan2(sqrt(1 - ((l - lsq) / (2 * self.l1 * self.l2)) ** 2), \
                          (l - lsq) / (2 * self.l1 * self.l2))

        return JointSpacePoint(theta1, theta2, 0, 0)

    def get_detailed_pos(self):
        """
        Returns origin, position of end of link 1, position of end of link 2
        """
        x1 = self.flip_x * self.l1 * cos(self.joints.theta1) + self.origin.x
        y1 = self.l1 * sin(self.joints.theta1) + self.origin.y

        return self.origin, Vector2D(x1, y1), Vector2D(self.tool.x, self.tool.y)

    def compute_jacobian(self):
        """
        Returns jacobian matrix at current state
        """
        dx_dth1 = - self.l1 * sin(self.joints.theta1) \
                  - self.l2 * sin(self.joints.theta1 + self.joints.theta2)
        dx_dth2 = - self.l2 * sin(self.joints.theta1 + self.joints.theta2)

        dy_dth1 = self.l1 * cos(self.joints.theta1) \
                  + self.l2 * cos(self.joints.theta1 + self.joints.theta2)
        dy_dth2 = self.l2 * cos(self.joints.theta1 + self.joints.theta2)

        return np.matrix([[dx_dth1, dx_dth2], \
                          [dy_dth1, dy_dth2]])

    def get_tool_vel(self, joints_vel):
        """
        Computes current tool velocity using jacobian
        """
        joints_vel = np.matrix([[joints_vel.theta1],
                                [joints_vel.theta2]])
        jacobian = self.compute_jacobian()
        tool_vel = jacobian * joints_vel

        return RobotSpacePoint(tool_vel[0], tool_vel[1], 0, 0)

    def get_joints_vel(self, tool_vel):
        """
        Computes current tool velocity using jacobian
        """
        tool_vel = np.matrix([[tool_vel.x], [tool_vel.y]])
        jacobian = self.compute_jacobian()

        if abs(np.linalg.det(jacobian)) < EPSILON:
            raise ValueError('Singularity')

        joints_vel = np.linalg.solve(jacobian, tool_vel)

        return JointSpacePoint(joints_vel[0], joints_vel[1], 0, 0)

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
        start_joints_vel = self.get_joints_vel(start_vel)

        target_joints_pos = self.inverse_kinematics(target_pos)
        target_joints_vel = self.get_joints_vel(target_vel)

        # Get synchronisation time
        tf_sync = self.synchronisation_time(start_joints_pos,
                                            start_joints_vel,
                                            target_joints_pos,
                                            target_joints_vel)

        # Get trajectories for each joint
        traj_theta1 = self.get_path_joint('theta1',
                                          start_joints_pos.theta1,
                                          start_joints_vel.theta1,
                                          target_joints_pos.theta1,
                                          target_joints_vel.theta1,
                                          tf_sync,
                                          delta_t)

        traj_theta2 = self.get_path_joint('theta2',
                                          start_joints_pos.theta2,
                                          start_joints_vel.theta2,
                                          target_joints_pos.theta2,
                                          target_joints_vel.theta2,
                                          tf_sync,
                                          delta_t)

        return traj_theta1, traj_theta2

    def get_path_joint(self, axis, pos_i, vel_i, pos_f, vel_f, tf_sync, delta_t):
        """
        Generates a time optimal trajectory for the specified joint
        Input:
        axis  - joint
        pos_i - initial position
        vel_i - initial velocity
        pos_f - final position
        vel_f - final velocity
        tf_sync - duration of the trajectory (synchronisation time)
        """
        # Get axis constraints
        constraint = self.constraints.get_axis_constraints(axis)

        # Compute limit time
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(constraint, pos_i, vel_i, pos_f, vel_f)

        if vel_i == 0 and vel_f == 0:
            vel_c = EPSILON
        elif abs(vel_i) > abs(vel_f):
            vel_c = vel_i
        else:
            vel_c = vel_f

        tf_lim = (delta_p / vel_c) \
                 + (0.5 * sign_traj * vel_c / constraint.acc_max)

        # Determine shape of trajectory
        if tf_sync < tf_lim or (vel_i == 0 and vel_f == 0):
            traj = self.trapezoidal_profile(constraint, pos_i, vel_i, pos_f, vel_f,
                                            tf_sync, tf_lim, delta_t)
        else:
            traj = self.doubleramp_profile(constraint, pos_i, vel_i, pos_f, vel_f,
                                           tf_sync, tf_lim, delta_t)

        return traj

    def trapezoidal_profile(self, constraint, pos_i, vel_i, pos_f, vel_f,
                            tf_sync, tf_lim, delta_t):
        """
        Generate a trapezoidal profile to reach target
        """
        # Compute cruise speed using equation 30
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(constraint, pos_i, vel_i, pos_f, vel_f)
        b = constraint.acc_max * tf_sync + sign_traj * vel_i

        vel_c = 0.5 * (b - sqrt(b**2 \
                                - 4 * sign_traj * constraint.acc_max * delta_p \
                                - 2 * (vel_i - vel_f)**2))

        return self.generic_profile(constraint, pos_i, vel_i, pos_f, vel_f,
                                    tf_sync, tf_lim, delta_t, sign_traj, vel_c)

    def doubleramp_profile(self, constraint, pos_i, vel_i, pos_f, vel_f,
                           tf_sync, tf_lim, delta_t):
        """
        Generate a double ramp profile to reach target
        """
        # Compute cruise speed using equation 31
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(constraint, pos_i, vel_i, pos_f, vel_f)

        vel_c = (sign_traj * delta_p \
                 - 0.5 * ((vel_i - vel_f)**2 / constraint.acc_max)) \
                / (tf_sync - ((vel_i  - vel_f) / (sign_traj * constraint.acc_max)))

        return self.generic_profile(constraint, pos_i, vel_i, pos_f, vel_f,
                                    tf_sync, tf_lim, delta_t, sign_traj, vel_c)

    def generic_profile(self, constraint, pos_i, vel_i, pos_f, vel_f,
                        tf_sync, tf_lim, delta_t, sign_traj, vel_c):
        """
        Generate a generic profile (valid for trapezoidal and double ramp)
        """
        # Equation 35
        sign_sync = np.sign(tf_lim - tf_sync)

        t1 = abs((vel_c - vel_i) / constraint.acc_max)

        t2 = tf_sync - abs((vel_c - vel_f) / constraint.acc_max)

        # First piece
        a0 = float(pos_i)
        a1 = float(vel_i)
        a2 = float(0.5 * sign_traj * constraint.acc_max)

        time_1, traj_pos_1, traj_vel_1, traj_acc_1 = \
            self.polynomial_piece_profile([a2, a1, a0], 0, t1, delta_t)

        # Second piece
        a0 = float(np.polyval([a2, a1, a0], t1))
        a1 = float(vel_c)
        a2 = float(0)

        time_2, traj_pos_2, traj_vel_2, traj_acc_2 = \
            self.polynomial_piece_profile([a2, a1, a0], t1, t2, delta_t)

        # Third piece
        a0 = float(np.polyval([a2, a1, a0], t2 - t1))
        a1 = float(vel_c)
        a2 = float(- 0.5 * sign_traj * constraint.acc_max)

        time_3, traj_pos_3, traj_vel_3, traj_acc_3 = \
            self.polynomial_piece_profile([a2, a1, a0], t2, tf_sync + delta_t,
                                          delta_t)

        # Combine piecewise trajectory
        time = np.concatenate((time_1, time_2, time_3), axis=0)
        traj_pos = np.concatenate((traj_pos_1, traj_pos_2, traj_pos_3), axis=0)
        traj_vel = np.concatenate((traj_vel_1, traj_vel_2, traj_vel_3), axis=0)
        traj_acc = np.concatenate((traj_acc_1, traj_acc_2, traj_acc_3), axis=0)

        return zip(time, traj_pos, traj_vel, traj_acc)

    def polynomial_piece_profile(self, polynome, start, stop, delta):
        """
        Generate a polynomial piece profile
        Return time, position, velocity and acceleration discrete profile
        """
        if stop < start:
            raise ValueError('Non causal trajectory profile requested')

        polynome_dot = np.polyder(polynome)
        polynome_dot_dot = np.polyder(polynome_dot)

        time = np.arange(start=start, stop=stop, step=delta, dtype=np.float32)
        dtime = np.arange(start=0, stop=stop-start, step=delta, dtype=np.float32)
        pos = np.polyval(polynome, dtime)
        vel = np.polyval(polynome_dot, dtime)
        acc = np.polyval(polynome_dot_dot, dtime)

        return time, pos, vel, acc

    def synchronisation_time(self, start_pos, start_vel, target_pos, target_vel):
        """
        Return largest time to destination to use slowest joint as synchronisation
        reference
        """
        # Compute time to destination for all joints
        ttd_theta1 = self.joint_time_to_destination('theta1',
                                                    start_pos.theta1,
                                                    start_vel.theta1,
                                                    target_pos.theta1,
                                                    target_vel.theta1)

        ttd_theta2 = self.joint_time_to_destination('theta2',
                                                    start_pos.theta2,
                                                    start_vel.theta2,
                                                    target_pos.theta2,
                                                    target_vel.theta2)

        # Return the largest one
        return np.amax([ttd_theta1.tf, ttd_theta2.tf])

    def joint_time_to_destination(self, axis, pos_i, vel_i, pos_f, vel_f):
        """
        Implements formula 27 in paper to compute minimal time to destination
        There is a mistake on the equation of tf on the paper: you need to
        substract the fraction from t2 instead of adding it (see eq 23)
        """
        if not self.constraints.trajectory_is_feasible(axis,
                                                       pos_i, vel_i,
                                                       pos_f, vel_f):
            raise

        constraint = self.constraints.get_axis_constraints(axis)

        delta_p = pos_f - pos_i

        if delta_p == 0:
            return TimeToDestination(0,0,0)

        sign_traj = self.trajectory_sign(constraint, pos_i, vel_i, pos_f, vel_f)

        t_1 = (sign_traj * constraint.vel_max - vel_i) \
              / (sign_traj * constraint.acc_max)

        t_2 = (1 / constraint.vel_max) \
              * ((vel_f**2 + vel_i**2 - 2 * sign_traj * vel_i) \
                 / (2 * constraint.acc_max) + (sign_traj * delta_p))

        t_f = t_2 - (vel_f - sign_traj * constraint.vel_max) \
                    / (sign_traj * constraint.acc_max)

        time_to_dest = TimeToDestination(t_1, t_2, t_f)

        return time_to_dest

    def trajectory_sign(self, constraint, pos_i, vel_i, pos_f, vel_f):
        """
        Get sign of trajectory to be executed
        """
        delta_p = pos_f - pos_i
        delta_v = vel_f - vel_i

        delta_p_crit = 0.5 * np.sign(delta_v) * (vel_f ** 2 - vel_i ** 2) \
                       / constraint.acc_max

        return np.sign(delta_p - delta_p_crit)
