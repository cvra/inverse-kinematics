from invkin import Scara
from math import pi, sqrt, cos, sin
import unittest

l1 = 1.0
l2 = 1.0

class ScaraTestCase(unittest.TestCase):
    def test_fwdkin_second_link(self):
        """
        Checks that forward kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0)

        x, y = scara.forward_kinematics()
        self.assertAlmostEqual(x, l1 + l2)
        self.assertAlmostEqual(y, 0.0)

        th2 = -pi / 2
        x, y = scara.update_joints(0.0, th2)
        self.assertAlmostEqual(x, l1 + l2 * cos(th2))
        self.assertAlmostEqual(y, l2 * sin(th2))

        th2 = pi / 4
        x, y = scara.update_joints(0.0, th2)
        self.assertAlmostEqual(x, l1 + l2 * cos(th2))
        self.assertAlmostEqual(y, l2 * sin(th2))

    def test_fwdkin_both_links(self):
        """
        Checks that forward kinematics works when both links are moving
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0)

        th1 = pi / 2
        th2 = pi / 2
        x, y = scara.update_joints(th1, th2)
        self.assertAlmostEqual(x, -l2)
        self.assertAlmostEqual(y, l1)

        th1 = -pi / 2
        th2 = pi / 2
        x, y = scara.update_joints(th1, th2)
        self.assertAlmostEqual(x, l2)
        self.assertAlmostEqual(y, -l1)


    def test_invkin_second_link(self):
        """
        Checks that inverse kinematics works if only the second link is moving
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0)

        x = l1 + l2
        y = 0.0
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, 0.0)
        self.assertAlmostEqual(th2, 0.0)

        x = (l1 + l2) / 2
        y = (l1 + l2) / 2
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, 0.0)
        self.assertAlmostEqual(th2, pi / 2)

    def test_invkin_both_links(self):
        """
        Checks that inverse kinematics works when both links move
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0)

        x = (l1 + l2) / 2
        y = -(l1 + l2) / 2
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, -pi / 2)
        self.assertAlmostEqual(th2, pi / 2)

        x = 0
        y = l1 + l2
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, pi / 2)
        self.assertAlmostEqual(th2, 0.0)

    def test_fwdkin_flip(self):
        """
        Checks that forward kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0, flip_x=-1)

        th2 = pi / 4
        x, y = scara.update_joints(0.0, th2)
        self.assertAlmostEqual(x, -(l1 + l2 * cos(th2)))
        self.assertAlmostEqual(y, l2 * sin(th2))

        th1 = pi / 2
        th2 = pi / 2
        x, y = scara.update_joints(th1, th2)
        self.assertAlmostEqual(x, l2)
        self.assertAlmostEqual(y, l1)

        th1 = -pi / 2
        th2 = pi / 2
        x, y = scara.update_joints(th1, th2)
        self.assertAlmostEqual(x, -l2)
        self.assertAlmostEqual(y, -l1)

    def test_invkin_flip(self):
        """
        Checks that inverse kinematics works with vertical flip
        """
        scara = Scara.Scara(l1=l1, l2=l2, theta1=0.0, theta2=0.0, flip_x=-1)

        x = -(l1 + l2) / 2
        y = -(l1 + l2) / 2
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, -pi / 2)
        self.assertAlmostEqual(th2, pi / 2)

        x = 0
        y = l1 + l2
        th1, th2 = scara.update_tool(x, y)
        self.assertAlmostEqual(th1, pi / 2)
        self.assertAlmostEqual(th2, 0.0)
