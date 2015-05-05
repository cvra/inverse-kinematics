from invkin.Datatypes import *
from invkin.DebraArm import DebraArm
from math import pi, cos, sin
import numpy as np

def get_path_joint_space(arm,
                         start_pose,
                         start_vel,
                         target_pose,
                         target_vel,
                         duration,
                         delta_t):
    "Moves a DebraArm by trajectory planning in joint space"
    start_joints = arm.update_tool(start_pose)
    target_joints = arm.update_tool(target_pose)

    traj_theta1 = move_joint(start_joints.theta1, 0,
                             target_joints.theta1, 0,
                             duration, delta_t)
    traj_theta2 = move_joint(start_joints.theta2, 0,
                             target_joints.theta2, 0,
                             duration, delta_t)
    traj_z = move_joint(start_joints.z, 0,
                        target_joints.z, 0,
                        duration, delta_t)
    traj_theta3 = move_joint(start_joints.theta3, 0,
                             target_joints.theta3, 0,
                             duration, delta_t)

    return traj_theta1, traj_theta2, traj_z, traj_theta3

def get_path_robot_space(arm,
                         start_pose,
                         start_vel,
                         target_pose,
                         target_vel,
                         duration,
                         delta_t):
    "Moves a DebraArm by trajectory planning in robot space"
    traj_x, traj_y = get_path_tool(arm,
                                   start_pose,
                                   start_vel,
                                   target_pose,
                                   target_vel,
                                   duration,
                                   delta_t)

    traj_theta1 = []
    traj_theta2 = []
    traj_z = []
    traj_theta3 = []

    for x, y in zip(traj_x, traj_y):
        tool = RobotSpacePoint(x[1], y[1], 0, 0)
        joints = arm.update_tool(tool)

        traj_theta1.append([x[0], joints.theta1, 0, 0])
        traj_theta2.append([x[0], joints.theta2, 0, 0])
        traj_z.append([x[0], joints.z, 0, 0])
        traj_theta3.append([x[0], joints.theta3, 0, 0])

    return traj_theta1, traj_theta2, traj_z, traj_theta3

def get_path_tool(arm,
                  start_pose,
                  start_vel,
                  target_pose,
                  target_vel,
                  duration,
                  delta_t):

    traj_x = move_joint(start_pose.x, start_vel.x,
                        target_pose.x, target_vel.x,
                        duration, delta_t)
    traj_y = move_joint(start_pose.y, start_vel.y,
                        target_pose.y, target_vel.y,
                        duration, delta_t)

    return traj_x, traj_y

def move_joint(start_pos,
               start_vel,
               target_pos,
               target_vel,
               duration,
               delta_t):
    "Moves one joint with nice ramps"
    # theta(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
    a0 = start_pos
    a1 = start_vel
    a2 = (3 * (target_pos - start_pos) / duration - (2 * start_vel + target_vel)) \
         / duration
    a3 = (-2 * (target_pos - start_pos) / duration + (start_vel + target_vel)) \
         / duration**2

    time = np.arange(start=0, stop=duration, step=delta_t, dtype=np.float32)

    traj_pos = np.polyval([a3, a2, a1, a0], time)
    traj_vel = np.polyval([3*a3, 2*a2, a1], time)
    traj_acc = np.polyval([6*a3, 2*a2], time)

    points = zip(time, traj_pos, traj_vel, traj_acc)

    return points
