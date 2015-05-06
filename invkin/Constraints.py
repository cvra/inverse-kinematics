from invkin.Datatypes import *
import numpy as np

class Constraints(object):
    "Arm constraints class"

    def __init__(self, th1_c=JointMinMaxConstraint(-1,1, 0,1, 0,1),
                       th2_c=JointMinMaxConstraint(-1,1, 0,1, 0,1),
                       z_c=JointMinMaxConstraint(-1,1, 0,1, 0,1),
                       th3_c=JointMinMaxConstraint(-1,1, 0,1, 0,1)):
        """
        Input:
        th1_c - constraints on theta1
        th2_c - constraints on theta2
        z_c - constraints on z
        th3_c - constraints on theta3
        """
        self.th1_c = th1_c
        self.th2_c = th2_c
        self.z_c = z_c
        self.th3_c = th3_c
