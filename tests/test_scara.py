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

        th1 = -pi / 2
        th2 = pi / 2
        x, y = scara.update_joints(th1, th2)
        self.assertAlmostEqual(x, l2)
        self.assertAlmostEqual(y, -l1)
