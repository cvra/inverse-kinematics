from invkin.Datatypes import *
from invkin import Joint
import unittest

class JointTestCase(unittest.TestCase):
    def test_traj_feasible(self):
        """
        Check that trajectory feasibility is well determined
        """
        joint = Joint.Joint()

        # Position too far
        with self.assertRaises(ValueError):
            feasible = joint.trajectory_is_feasible(0, 0, 2, 0)

        # Velocity too high
        with self.assertRaises(ValueError):
            feasible = joint.trajectory_is_feasible(0, 0, 0.5, 2)

        # Doable
        feasible = joint.trajectory_is_feasible(0, 0, 0.5, 0.5)
        self.assertAlmostEqual(feasible, TRUE)

    def test_traj_sign(self):
        """
        Check that trajectory sign is well determined
        """
        joint = Joint.Joint()

        # Positive
        sign = joint.trajectory_sign(0, 0, 0.5, 0.5)
        self.assertAlmostEqual(sign, 1)

        # Negative
        sign = joint.trajectory_sign(0, 0, -0.5, 0.5)
        self.assertAlmostEqual(sign, -1)

        # Non feasible
        sign = joint.trajectory_sign(0, 0, 0.5, 1)
        self.assertAlmostEqual(sign, 0)

    def test_time_to_destination(self):
        """
        Check that time to destination is well computed
        """
        joint = Joint.Joint()

        # Position too far
        with self.assertRaises(ValueError):
            ttd = joint.time_to_destination(0, 0, 2, 0)

        # Velocity too high
        with self.assertRaises(ValueError):
            ttd = joint.time_to_destination(0, 0, 0.5, 2)

        ttd = joint.time_to_destination(0, 0, 0.5, 0)
        self.assertAlmostEqual(ttd.t1, 1.0)
        self.assertAlmostEqual(ttd.t2, 0.5)
        self.assertAlmostEqual(ttd.tf, 1.5)

        ttd = joint.time_to_destination(0, 0, -0.75, 0)
        self.assertAlmostEqual(ttd.t1, 1.0)
        self.assertAlmostEqual(ttd.t2, 0.75)
        self.assertAlmostEqual(ttd.tf, 1.75)

        ttd = joint.time_to_destination(0, 0, 0.5, 0.5)
        self.assertAlmostEqual(ttd.t1, 1.0)
        self.assertAlmostEqual(ttd.t2, 0.625)
        self.assertAlmostEqual(ttd.tf, 1.125)

        ttd = joint.time_to_destination(0, 0, 0.75, 0.5)
        self.assertAlmostEqual(ttd.t1, 1.0)
        self.assertAlmostEqual(ttd.t2, 0.875)
        self.assertAlmostEqual(ttd.tf, 1.375)

        ttd = joint.time_to_destination(0, 0, -0.75, 0.5)
        self.assertAlmostEqual(ttd.t1, 1.0)
        self.assertAlmostEqual(ttd.t2, 0.875)
        self.assertAlmostEqual(ttd.tf, 2.375)

if __name__ == '__main__':
    unittest.main()
