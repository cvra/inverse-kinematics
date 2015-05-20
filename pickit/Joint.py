from __future__ import division
from pickit.Datatypes import *
import numpy as np

class Joint(object):
    "Robot generic joint class"

    def __init__(self, name, constraints=JointMinMaxConstraint(-1,1, -1,1, -1,1)):
        self.name = name
        self.constraints = constraints

    def trajectory_is_feasible(self, pos_i, vel_i, pos_f, vel_f):
        """
        Implements formula 9 and checks boundaries to determine feasibility
        """
        constraint = self.constraints

        if pos_f > (constraint.pos_max + EPSILON) \
           or pos_f < (constraint.pos_min - EPSILON):
            raise ValueError('Target position unreachable by joint ' + self.name)
        if vel_f > (constraint.vel_max + EPSILON) \
           or vel_f < (constraint.vel_min) - EPSILON:
            raise ValueError('Target velocity unreachable by joint ' + self.name)

        delta_p_dec = 0.5 * vel_f * abs(vel_f) / constraint.acc_max

        if (pos_f + delta_p_dec) > (constraint.pos_max + EPSILON) \
           or (pos_f + delta_p_dec) < (constraint.pos_min - EPSILON):
           raise ValueError('Target position unreachable at specified velocity by joint ' + self.name)

        return TRUE

    def get_path(self, pos_i, vel_i, pos_f, vel_f, tf_sync, delta_t):
        """
        Generates a time optimal trajectory
        Input:
        pos_i - initial position
        vel_i - initial velocity
        pos_f - final position
        vel_f - final velocity
        tf_sync - duration of the trajectory (synchronisation time)
        """
        # Get axis constraints
        constraint = self.constraints

        # Compute limit time
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(pos_i, vel_i, pos_f, vel_f)

        if abs(vel_i) < EPSILON and abs(vel_f) < EPSILON:
            vel_c = EPSILON # ensure we use a trapezoidal trajectory
        elif abs(vel_i) < abs(vel_f):
            vel_c = vel_f
        else:
            vel_c = vel_i

        tf_lim = (delta_p / vel_c) \
                 + (0.5 * sign_traj * (vel_c - vel_f) / constraint.acc_max) \
                 + (0.5 * sign_traj * (vel_i - vel_c) / constraint.acc_max)

        # Determine shape of trajectory
        if tf_sync < tf_lim or (vel_i == 0 and vel_f == 0):
            traj = self.trapezoidal_profile(pos_i, vel_i, pos_f, vel_f,
                                            tf_sync, tf_lim, delta_t)
        else:
            traj = self.doubleramp_profile(pos_i, vel_i, pos_f, vel_f,
                                           tf_sync, tf_lim, delta_t)

        path = []
        for (t, pos, vel, acc) in traj:
            path.append((t, pos, vel, acc))

        return path

    def trapezoidal_profile(self, pos_i, vel_i, pos_f, vel_f,
                            tf_sync, tf_lim, delta_t):
        """
        Generate a trapezoidal profile to reach target
        """
        constraint = self.constraints

        # Compute cruise speed using equation 30
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(pos_i, vel_i, pos_f, vel_f)
        # Avoid division by 0
        if sign_traj == 0:
            sign_traj = 1

        b = constraint.acc_max * tf_sync + sign_traj * vel_i

        vel_c = 0.5 * (b - np.sqrt(b**2 \
                                - 4 * sign_traj * constraint.acc_max * delta_p \
                                - 2 * (vel_i - vel_f)**2))

        return self.generic_profile(pos_i, vel_i, pos_f, vel_f,
                                    tf_sync, tf_lim, delta_t, sign_traj, 1, vel_c)

    def doubleramp_profile(self, pos_i, vel_i, pos_f, vel_f,
                           tf_sync, tf_lim, delta_t):
        """
        Generate a double ramp profile to reach target
        """
        constraint = self.constraints

        # Compute cruise speed using equation 31
        delta_p = pos_f - pos_i
        sign_traj = self.trajectory_sign(pos_i, vel_i, pos_f, vel_f)
        # Avoid division by 0
        if sign_traj == 0:
            sign_traj = 1

        vel_c = (sign_traj * delta_p \
                 - 0.5 * ((vel_i - vel_f)**2 / constraint.acc_max)) \
                / (tf_sync - ((vel_i  - vel_f) / (sign_traj * constraint.acc_max)))

        return self.generic_profile(pos_i, vel_i, pos_f, vel_f,
                                    tf_sync, tf_lim, delta_t, sign_traj, 1, vel_c)

    def generic_profile(self, pos_i, vel_i, pos_f, vel_f,
                        tf_sync, tf_lim, delta_t, sign_traj, sign_sync, vel_c):
        """
        Generate a generic profile (valid for trapezoidal and double ramp)
        """
        constraint = self.constraints

        # Equation 35
        t1 = (sign_traj * vel_c - vel_i) / (sign_traj * constraint.acc_max)
        t2 = tf_sync - abs(vel_c - vel_f) / constraint.acc_max

        # First piece
        a0 = float(pos_i)
        a1 = float(vel_i)
        a2 = float(0.5 * sign_traj * sign_sync * constraint.acc_max)

        time_1, traj_pos_1, traj_vel_1, traj_acc_1 = \
            self.polynomial_piece_profile([a2, a1, a0], 0, t1, delta_t)

        # Second piece
        a0 = float(np.polyval([a2, a1, a0], t1))
        a1 = float(sign_traj * vel_c)
        a2 = float(0)

        time_2, traj_pos_2, traj_vel_2, traj_acc_2 = \
            self.polynomial_piece_profile([a2, a1, a0], t1, t2, delta_t)

        # Third piece
        a0 = float(np.polyval([a2, a1, a0], t2 - t1))
        a1 = float(sign_traj * vel_c)
        a2 = float(- 0.5 * sign_traj * constraint.acc_max)

        time_3, traj_pos_3, traj_vel_3, traj_acc_3 = \
            self.polynomial_piece_profile([a2, a1, a0], t2, tf_sync, delta_t)

        time_4 = [tf_sync]
        traj_pos_4 = [pos_f]
        traj_vel_4 = [vel_f]
        traj_acc_4 = [0.0]

        # Combine piecewise trajectory
        time = np.concatenate((time_1, time_2, time_3, time_4), axis=0)
        traj_pos = \
            np.concatenate((traj_pos_1, traj_pos_2, traj_pos_3, traj_pos_4), axis=0)
        traj_vel = \
            np.concatenate((traj_vel_1, traj_vel_2, traj_vel_3, traj_vel_4), axis=0)
        traj_acc = \
            np.concatenate((traj_acc_1, traj_acc_2, traj_acc_3, traj_acc_4), axis=0)

        time = time.tolist()
        traj_pos = traj_pos.tolist()
        traj_vel = traj_vel.tolist()
        traj_acc = traj_acc.tolist()

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

    def time_to_destination(self, pos_i, vel_i, pos_f, vel_f):
        """
        Implements formula 27 in paper to compute minimal time to destination
        There is a mistake on the equation of tf on the paper: you need to
        substract the fraction from t2 instead of adding it (see eq 23)
        """
        if not self.trajectory_is_feasible(pos_i, vel_i, pos_f, vel_f):
            raise

        constraint = self.constraints

        delta_p = pos_f - pos_i

        if delta_p == 0:
            return TimeToDestination(0,0,0)

        sign_traj = self.trajectory_sign(pos_i, vel_i, pos_f, vel_f)

        t_1 = (sign_traj * constraint.vel_max - vel_i) \
              / (sign_traj * constraint.acc_max)

        t_2 = (1 / constraint.vel_max) \
              * ((vel_f**2 + vel_i**2 - 2 * sign_traj * vel_i) \
                 / (2 * constraint.acc_max) + (sign_traj * delta_p))

        t_f = t_2 - (vel_f - sign_traj * constraint.vel_max) \
                    / (sign_traj * constraint.acc_max)

        time_to_dest = TimeToDestination(t_1, t_2, t_f)

        return time_to_dest

    def trajectory_sign(self, pos_i, vel_i, pos_f, vel_f):
        """
        Get sign of trajectory to be executed
        """
        constraint = self.constraints

        delta_p = pos_f - pos_i
        delta_v = vel_f - vel_i

        delta_p_crit = 0.5 * np.sign(delta_v) * (vel_f ** 2 - vel_i ** 2) \
                       / constraint.acc_max

        return np.sign(delta_p - delta_p_crit)

    def set_constraints(self, new_constraints):
        self.constraints = new_constraints

    def get_constraints(self):
        return self.constraints
