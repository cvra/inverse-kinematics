import pygame, sys
from invkin.Scara import *
from invkin.Datatypes import *
from math import cos, sin

PX_PER_METER = 100
L1 = 1.0
L2 = 0.5
WIDTH = int(2 * (L1 + L2) * PX_PER_METER)
HEIGHT = int(2 * (L1 + L2) * PX_PER_METER)

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

    scara = Scara(l1=L1, l2=L2, flip_x=-1)
    tool = scara.forward_kinematics()
    joints = scara.inverse_kinematics()

    # Draw robot
    p0, p1, p2 = scara.get_detailed_pos()
    draw_scara(p0, p1, p2)

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
            print("cursor: x: ", x, ", y: ", y)

            try:
                tool = RobotSpacePoint(x, y, 0, 0)
                joints = scara.update_tool(tool)
            except ValueError:
                pass

            print("scara: ", "x:", scara.tool.x, "y:", scara.tool.y, \
                  "th1:", scara.joints.theta1, "th2:", scara.joints.theta2)

            # Draw robot
            origin, p1, p2 = scara.get_detailed_pos()
            draw_scara(origin, p1, p2)

            pygame.display.update()

def draw_scara(p0, p1, p2):
    "draw scara state"

    draw_line(p0.x, p0.y, p1.x, p1.y)
    draw_line(p1.x, p1.y, p2.x, p2.y)

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
