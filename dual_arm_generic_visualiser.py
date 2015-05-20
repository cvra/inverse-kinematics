import pygame, sys, time
import matplotlib.pyplot as plt
from pickit.Datatypes import *
from pickit import DebraArm, ArmManager
from math import cos, sin, pi

# Robot settings
L1 = 1.5
L2 = 1.0
L3 = 0.2
GRIPPER_HEADING = 0
RANGE_MIN = abs(L1 - L2)
RANGE_MAX = abs(L1 + L2)

# Trajectory generation settings
DELTA_T = 0.05

# Display settings
PX_PER_METER = 100
WIDTH = int(3 * (L1 + L2 + L3) * PX_PER_METER)
HEIGHT = int(3 * (L1 + L2 + L3) * PX_PER_METER)

pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
PURPLE = (255, 0, 255)

def main():
    "draw loop"
    ALTERNATE = 0

    # Initial robot state
    right_origin = Vector3D(0.5, 0.0, 0.0)
    left_origin = Vector3D(-0.5, 0.0, 0.0)

    right_arm = DebraArm.DebraArm(l1=L1, l2=L2, origin=right_origin, flip_x=1)
    right_arm.inverse_kinematics(RobotSpacePoint(0.99*(L1+L2) + right_origin.x,
                                                 0 + right_origin.y,
                                                 0 + right_origin.z,
                                                 0))
    tool = right_arm.get_tool()
    joints = right_arm.get_joints()

    left_arm = DebraArm.DebraArm(l1=L1, l2=L2, origin=left_origin, flip_x=-1)
    left_arm.inverse_kinematics(RobotSpacePoint(-0.99*(L1+L2) + left_origin.x,
                                                0 + left_origin.y,
                                                0 + left_origin.z,
                                                0))
    tool = left_arm.get_tool()
    joints = left_arm.get_joints()

    ws_front = Workspace(-1.5, 1.5,
                         abs(L1 - L2), abs(L1 + L2),
                         0.0, 0.2,
                         1)
    ws_back = Workspace(-1.5, 1.5,
                        -abs(L1 + L2), -abs(L1 - L2),
                        0.0, 0.2,
                        -1)
    ws_right = Workspace(abs(L1 - L2) + right_origin.x, abs(L1 + L2) + right_origin.x,
                        -1.5 + right_origin.y, 1.5 + right_origin.y,
                        0.0 + right_origin.z, 0.2 + right_origin.z,
                        1)
    ws_left = Workspace(-abs(L1 + L2) + left_origin.x, -abs(L1 - L2) + left_origin.x,
                        -1.5 + left_origin.y, 1.5 + left_origin.y,
                        0.0 + left_origin.z, 0.2 + left_origin.z,
                        1)

    right_arm_manager = ArmManager.ArmManager(right_arm, ws_front, ws_right, ws_back, DELTA_T)
    left_arm_manager = ArmManager.ArmManager(left_arm, ws_front, ws_left, ws_back, DELTA_T)

    # Draw right arm
    origin, p1, p2, p3, z = right_arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
    draw_workspaces(ws_front, ws_right, ws_back)
    # Draw left arm
    origin, p1, p2, p3, z = left_arm.get_detailed_pos(L3)
    draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
    draw_workspaces(ws_front, ws_left, ws_back)

    pygame.display.update()

    paused = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = get_cursor_pos()
                if ALTERNATE:
                    ALTERNATE = 0
                    tool_prev = right_arm_manager.arm.get_tool()
                    tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                    start_time = time.time()
                    pth1, pth2, pz, pth3 = right_arm_manager.goto(tool_prev, RobotSpacePoint(0,0,0,0),
                                                                  tool, RobotSpacePoint(0,0,0,0),
                                                                  'line')
                    elapsed_time = time.time() - start_time
                    print('elapsed time: ', elapsed_time)

                    graph_trajectory_joint(pth1, pth2, pth3)
                    draw_trajectory(right_arm_manager.arm, pth1, pth2, pz, pth3, DELTA_T)

                else:
                    ALTERNATE = 1
                    tool_prev = left_arm_manager.arm.get_tool()
                    tool = RobotSpacePoint(x, y, z, GRIPPER_HEADING)

                    start_time = time.time()
                    pth1, pth2, pz, pth3 = left_arm_manager.goto(tool_prev, RobotSpacePoint(0,0,0,0),
                                                                 tool, RobotSpacePoint(0,0,0,0),
                                                                 'line')
                    elapsed_time = time.time() - start_time
                    print('elapsed time: ', elapsed_time)

                    graph_trajectory_joint(pth1, pth2, pth3)
                    draw_trajectory(left_arm_manager.arm, pth1, pth2, pz, pth3, DELTA_T)

        if not paused:
            SCREEN.fill(BLACK)

            # Draw right arm
            origin, p1, p2, p3, z = right_arm_manager.arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
            draw_workspaces(ws_front, ws_right, ws_back)
            # Draw left arm
            origin, p1, p2, p3, z = left_arm_manager.arm.get_detailed_pos(L3)
            draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)
            draw_workspaces(ws_front, ws_left, ws_back)

            pygame.display.update()

def graph_trajectory_xyz(px, py, pz, pgrp):
    graph_axis_trajectory(px, 'x')
    graph_axis_trajectory(py, 'y')
    graph_axis_trajectory(pz, 'z')
    graph_axis_trajectory(pgrp, 'grp')

def graph_trajectory_joint(pth1, pth2, pth3):
    graph_axis_trajectory(pth1, 'th1')
    graph_axis_trajectory(pth2, 'th2')
    graph_axis_trajectory(pth3, 'th3')

def graph_axis_trajectory(axis, pdf_name):
    time = []
    pos = []
    vel = []
    acc = []
    for t in axis:
        time.append(t[0])
        pos.append(t[1])
        vel.append(t[2])
        acc.append(t[3])

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(311)
    ax1.plot(time, pos)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax2 = fig.add_subplot(312)
    ax2.plot(time, vel)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax3 = fig.add_subplot(313)
    ax3.plot(time, acc)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s^2)')
    fig.savefig((pdf_name + ".pdf"))

def draw_trajectory(arm, path_th1, path_th2, path_z, path_th3, dt):
    "draw trajectory"

    for th1, th2, z, th3 in zip(path_th1, path_th2, path_z, path_th3):
        joints = JointSpacePoint(th1[1], th2[1], z[1], th3[1])
        tool = arm.forward_kinematics(joints)
        get_robot_new_state(arm, tool)

        # print("arm: ", "x:", arm.tool.x, "y:", arm.tool.y, \
        #       "th1:", arm.joints.theta1, "th2:", arm.joints.theta2)

        # Draw robot
        origin, p1, p2, p3, z = arm.get_detailed_pos(L3)
        draw_arm(origin, p1, p2, p3, RANGE_MIN, RANGE_MAX)

        pygame.display.update()

        time.sleep(dt)

def get_robot_new_state(arm, new_tool):
    "get robot's current state"
    try:
        return arm.inverse_kinematics(new_tool)
    except ValueError:
        return arm.get_joints()

def get_cursor_pos():
    "get cursor position"
    (x, y) = pygame.mouse.get_pos()
    x = (x - WIDTH/2) / PX_PER_METER
    y = - (y - HEIGHT/2) / PX_PER_METER
    print("cursor: x: ", x, ", y: ", y)

    return x, y

def draw_workspaces(ws_1, ws_2, ws_3):
    "draw workspaces"
    draw_workspace(ws_1)
    draw_workspace(ws_2)
    draw_workspace(ws_3)

def draw_workspace(workspace):
    "draw workspace"
    draw_line(workspace.x_min, workspace.y_min, workspace.x_max, workspace.y_min, GREEN)
    draw_line(workspace.x_max, workspace.y_min, workspace.x_max, workspace.y_max, GREEN)
    draw_line(workspace.x_max, workspace.y_max, workspace.x_min, workspace.y_max, GREEN)
    draw_line(workspace.x_min, workspace.y_max, workspace.x_min, workspace.y_min, GREEN)

def draw_arm(p0, p1, p2, p3, RANGE_MIN, RANGE_MAX):
    "draw arm state"

    draw_line(p0.x, p0.y, p1.x, p1.y, CYAN)
    draw_line(p1.x, p1.y, p2.x, p2.y, CYAN)
    draw_line(p2.x, p2.y, p3.x, p3.y, CYAN)
    draw_circle(p0.x, p0.y, RANGE_MAX)
    if RANGE_MIN * PX_PER_METER > 1:
        draw_circle(p0.x, p0.y, RANGE_MIN)

def draw_line(pos1_x, pos1_y, pos2_x, pos2_y, color):
    "draw line from pos1 to pos2"
    pygame.draw.line(SCREEN,
                     color,
                     (int(pos1_x * PX_PER_METER + WIDTH/2),
                      int(-pos1_y * PX_PER_METER + HEIGHT/2)),
                     (int(pos2_x * PX_PER_METER + WIDTH/2),
                      int(-pos2_y * PX_PER_METER + HEIGHT/2)),
                     2)

def draw_circle(pos_x, pos_y, radius):
    "draw circle from center position and radius"
    pygame.draw.circle(SCREEN,
                       RED,
                       (int(pos_x * PX_PER_METER + WIDTH/2),
                        int(-pos_y * PX_PER_METER + HEIGHT/2)),
                       int(radius * PX_PER_METER),
                       2)

if __name__ == "__main__":
    main()
