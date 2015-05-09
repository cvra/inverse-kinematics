from invkin.Datatypes import *
from invkin import Constraints
import unittest

class ConstraintsTestCase(unittest.TestCase):
    def test_add_axis(self):
        """
        Checks that adding an axis works
        """
        constraints = Constraints.Constraints()
        constraints.add_axis('theta1')
        constraint_theta1 = constraints.get_axis_constraints('theta1')

        self.assertAlmostEqual(constraint_theta1.pos_min, -1)
        self.assertAlmostEqual(constraint_theta1.pos_max,  1)
        self.assertAlmostEqual(constraint_theta1.vel_min,  0)
        self.assertAlmostEqual(constraint_theta1.vel_max,  1)
        self.assertAlmostEqual(constraint_theta1.acc_min,  0)
        self.assertAlmostEqual(constraint_theta1.acc_max,  1)

        constraints.add_axis('theta2', JointMinMaxConstraint(-5,5, 0.1,10, 0.2,100))
        constraint_theta2 = constraints.get_axis_constraints('theta2')

        self.assertAlmostEqual(constraint_theta2.pos_min,  -5)
        self.assertAlmostEqual(constraint_theta2.pos_max,   5)
        self.assertAlmostEqual(constraint_theta2.vel_min,   0.1)
        self.assertAlmostEqual(constraint_theta2.vel_max,  10)
        self.assertAlmostEqual(constraint_theta2.acc_min,   0.2)
        self.assertAlmostEqual(constraint_theta2.acc_max, 100)

    def test_traj_feasible(self):
        """
        Check that trajectory feasibility is well determined
        """
        constraints = Constraints.Constraints()
        constraints.add_axis('theta1')

        # Position too far
        with self.assertRaises(ValueError):
            feasible = constraints.trajectory_is_feasible('theta1', 0, 0, 2, 0)

        # Velocity too high
        with self.assertRaises(ValueError):
            feasible = constraints.trajectory_is_feasible('theta1', 0, 0, 0.5, 2)

        # Doable
        feasible = constraints.trajectory_is_feasible('theta1', 0, 0, 0.5, 0.5)
        self.assertAlmostEqual(feasible, TRUE)
