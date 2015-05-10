from invkin.Datatypes import *
from invkin.Joint import Joint
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

        self.theta1_axis = Joint(theta1_constraints)
        self.theta2_axis = Joint(theta2_constraints)

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
        traj_theta1 = self.theta1_axis.get_path(start_joints_pos.theta1,
                                                start_joints_vel.theta1,
                                                target_joints_pos.theta1,
                                                target_joints_vel.theta1,
                                                tf_sync,
                                                delta_t)

        traj_theta2 = self.theta2_axis.get_path(start_joints_pos.theta2,
                                                start_joints_vel.theta2,
                                                target_joints_pos.theta2,
                                                target_joints_vel.theta2,
                                                tf_sync,
                                                delta_t)

        return traj_theta1, traj_theta2

    def synchronisation_time(self, start_pos, start_vel, target_pos, target_vel):
        """
        Return largest time to destination to use slowest joint as synchronisation
        reference
        """
        # Compute time to destination for all joints
        ttd_theta1 = self.theta1_axis.time_to_destination(start_pos.theta1,
                                                          start_vel.theta1,
                                                          target_pos.theta1,
                                                          target_vel.theta1)

        ttd_theta2 = self.theta2_axis.time_to_destination(start_pos.theta2,
                                                          start_vel.theta2,
                                                          target_pos.theta2,
                                                          target_vel.theta2)

        # Return the largest one
        return np.amax([ttd_theta1.tf, ttd_theta2.tf])
