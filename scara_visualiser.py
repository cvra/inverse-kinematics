import pygame, sys
from invkin import Scara
from math import cos, sin

PX_PER_METER = 100
L1 = 1.0
L2 = 1.0
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

    scara = Scara.Scara(l1=L1, l2=L2, theta1=0.0, theta2=0.0, flip_x=-1)
    tool_x, tool_y = scara.forward_kinematics()
    th1, th2 = scara.inverse_kinematics()

    # Draw robot
    o_x, o_y, x1, y1, x2, y2 = scara.get_detailed_pos()
    draw_scara(o_x, o_y, x1, y1, x2, y2)

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

            th1, th2 = scara.update_tool(x, y)
            print("scara: ", "x:", scara.x, "y:", scara.y, "th1:", scara.theta1, "th2:", scara.theta2)

            # Draw robot
            o_x, o_y, x1, y1, x2, y2 = scara.get_detailed_pos()
            draw_scara(o_x, o_y, x1, y1, x2, y2)

            pygame.display.update()

def draw_scara(origin_x, origin_y, l1_x, l1_y, l2_x, l2_y):
    "draw scara state"

    draw_line(origin_x, origin_y, l1_x, l1_y)
    draw_line(l1_x, l1_y, l2_x, l2_y)

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
