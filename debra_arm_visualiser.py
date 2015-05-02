import pygame, sys
from invkin import DebraArm
from math import cos, sin, pi

PX_PER_METER = 100
L1 = 1.0
L2 = 1.0
L3 = 0.2
GRIPPER_HEADING = 0
WIDTH = int(2 * (L1 + L2 + L3) * PX_PER_METER)
HEIGHT = int(2 * (L1 + L2 + L3) * PX_PER_METER)

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

    # Initial robot state
    origin_x, origin_y = 0.0, 0.0

    arm = DebraArm.DebraArm(l1=L1, l2=L2, theta1=0, theta2=0, z=0, theta3=0, flip_x=1)
    tool_x, tool_y, tool_z, tool_hdg = arm.forward_kinematics()
    th1, th2, z, th3 = arm.inverse_kinematics()

    # Draw robot
    o_x, o_y, o_z, x1, y1, x2, y2, x3, y3, z = arm.get_detailed_pos(L3)
    draw_arm(o_x, o_y, x1, y1, x2, y2, x3, y3)

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

        if not paused:
            SCREEN.fill(BLACK)

            (x, y) = pygame.mouse.get_pos()
            x = (x - WIDTH/2) / PX_PER_METER
            y = - (y - HEIGHT/2) / PX_PER_METER
            print("x: ", x, ", y: ", y)

            th1, th2, z, th3 = arm.update_tool(x, y, z, GRIPPER_HEADING)
            print("arm: ", "x:", arm.x, "y:", arm.y, "th1:", arm.theta1, "th2:", arm.theta2)

            # Draw robot
            o_x, o_y, o_z, x1, y1, x2, y2, x3, y3, z = arm.get_detailed_pos(L3)
            draw_arm(o_x, o_y, x1, y1, x2, y2, x3, y3)

            pygame.display.update()

def draw_arm(origin_x, origin_y, l1_x, l1_y, l2_x, l2_y, l3_x, l3_y):
    "draw arm state"

    draw_line(origin_x, origin_y, l1_x, l1_y)
    draw_line(l1_x, l1_y, l2_x, l2_y)
    draw_line(l2_x, l2_y, l3_x, l3_y)

def draw_line(pos1_x, pos1_y, pos2_x, pos2_y):
    "draw line from pos1 to pos2"
    pygame.draw.line(SCREEN,
                     CYAN,
                     (int(pos1_x * PX_PER_METER + WIDTH/2),
                      int(-pos1_y * PX_PER_METER + HEIGHT/2)),
                     (int(pos2_x * PX_PER_METER + WIDTH/2),
                      int(-pos2_y * PX_PER_METER + HEIGHT/2)),
                     2)

if __name__ == "__main__":
    main()
