from invkin import DebraArm
from math import pi, sqrt, cos, sin
import unittest

l1 = 1.0
l2 = 0.5

class DebraArmTestCase(unittest.TestCase):
    def test_fwdkin(self):
        """
        Checks that forward kinematics works
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, theta1=0, theta2=0, z=0, theta3=0)

        th1 = pi / 2
        th2 = pi / 2
        z = 0.1
        th3 = pi / 2
        x, y, z, grp_hdg = arm.update_joints(th1, th2, z, th3)
        self.assertAlmostEqual(x, -l2)
        self.assertAlmostEqual(y, l1)
        self.assertAlmostEqual(z, 0.1)
        self.assertAlmostEqual(grp_hdg, -pi)

        th1 = -pi / 2
        th2 = pi / 2
        z = 0.2
        th3 = pi / 2
        x, y, z, grp_hdg = arm.update_joints(th1, th2, z, th3)
        self.assertAlmostEqual(x, l2)
        self.assertAlmostEqual(y, -l1)
        self.assertAlmostEqual(z, 0.2)
        self.assertAlmostEqual(grp_hdg, 0.0)

    def test_invkin(self):
        """
        Checks that inverse kinematics works
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, theta1=0.0, theta2=0.0)

        x = l2
        y = - l1
        z = 0.1
        grp_hdg = 0
        th1, th2, z, th3 = arm.update_tool(x, y, z, grp_hdg)
        self.assertAlmostEqual(th1, -pi / 2)
        self.assertAlmostEqual(th2, pi / 2)
        self.assertAlmostEqual(z, 0.1)
        self.assertAlmostEqual(th3, pi / 2)

        x = 0
        y = l1 + l2
        z = 0.2
        grp_hdg = 0
        th1, th2, z, th3 = arm.update_tool(x, y, z, grp_hdg)
        self.assertAlmostEqual(th1, pi / 2)
        self.assertAlmostEqual(th2, 0.0)
        self.assertAlmostEqual(z, 0.2)
        self.assertAlmostEqual(th3, 0)

    def test_fwdkin_origin(self):
        """
        Checks that forward kinematics works with origin not at 0
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, theta1=0, theta2=0, z=0, theta3 = 0,
                                origin=(1,1,1))

        th1 = pi / 2
        th2 = pi / 2
        z = 0.1
        th3 = pi / 2
        x, y, z, grp_hdg = arm.update_joints(th1, th2, z, th3)
        self.assertAlmostEqual(x, -l2 + 1)
        self.assertAlmostEqual(y, l1 + 1)
        self.assertAlmostEqual(z, 0.1 + 1)
        self.assertAlmostEqual(grp_hdg, -pi)

        th1 = -pi / 2
        th2 = pi / 2
        z = 0.2
        th3 = pi / 2
        x, y, z, grp_hdg = arm.update_joints(th1, th2, z, th3)
        self.assertAlmostEqual(x, l2 + 1)
        self.assertAlmostEqual(y, -l1 + 1)
        self.assertAlmostEqual(z, 0.2 + 1)
        self.assertAlmostEqual(grp_hdg, 0.0)

    def test_invkin_origin(self):
        """
        Checks that inverse kinematics works with origin not at 0
        """
        arm = DebraArm.DebraArm(l1=l1, l2=l2, theta1=0, theta2=0, z=0, theta3 = 0,
                                origin=(1,1,1))
        x = 1 + l2
        y = 1 - l1
        z = 0.1
        grp_hdg = 0
        th1, th2, z, th3 = arm.update_tool(x, y, z, grp_hdg)
        self.assertAlmostEqual(th1, -pi / 2)
        self.assertAlmostEqual(th2, pi / 2)
        self.assertAlmostEqual(z, 0.1)
        self.assertAlmostEqual(th3, pi / 2)

        x = 1
        y = 1 + l1 + l2
        z = 0.2
        th3 = pi / 2
        th1, th2, z, th3 = arm.update_tool(x, y, z, grp_hdg)
        self.assertAlmostEqual(th1, pi / 2)
        self.assertAlmostEqual(z, 0.2)
        self.assertAlmostEqual(th3, 0)
        self.assertAlmostEqual(th2, 0.0)
