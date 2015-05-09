from invkin.Datatypes import *
import numpy as np
import collections

class Constraints(collections.MutableMapping):
    "Arm constraints class"

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def add_axis(self, name, constraints=JointMinMaxConstraint(-1,1, 0,1, 0,1)):
        self.store[self.__keytransform__(name)] = constraints

    def get_axis_constraints(self, axis):
        return self.__getitem__(axis)

    def trajectory_is_feasible(self, axis, pos_i, vel_i, pos_f, vel_f):
        """
        Implements formula 9 and checks boundaries to determine feasibility
        """
        constraint = self.get_axis_constraints(axis)

        if pos_f > constraint.pos_max or pos_f < constraint.pos_min:
            raise ValueError('Target position unreachable')
        if vel_f > constraint.vel_max or vel_f < constraint.vel_min:
            raise ValueError('Target velocity unreachable')

        delta_p_dec = 0.5 * vel_f * abs(vel_f) / constraint.acc_max

        if (pos_f + delta_p_dec) > constraint.pos_max \
           or (pos_f + delta_p_dec) < constraint.pos_min:
           raise ValueError('Target position unreachable at specified velocity')

        return TRUE

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
