# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod
import math
import numpy as np


class Anneal(ABC):

    @abstractmethod
    def __init__(self, params=None):
        self.initialize_params(params)

        self.val = None

    @abstractmethod
    def anneal(self, current_step):
        pass

    @abstractmethod
    def initialize_params(self, params=None):
        if params is None:
            self.params = {}
        else:
            self.params = params

    def __float__(self):
        return self.val

    def __lt__(self, other):
        return self.val < other

    def __le__(self, other):
        return self.val <= other

    def __eq__(self, other):
        return self.val == other

    def __ne__(self, other):
        return self.val != other

    def __ge__(self, other):
        return self.val >= other

    def __gt__(self, other):
        return self.val > other

    def __neg__(self):
        return self.val * (-1)

    def __abs__(self):
        return abs(self.val)

    def __pos__(self):
        return self.val

    def __radd__(self, other):
        return self.val + other

    def __rsub__(self, other):
        return self.val - other

    def __rmul__(self, other):
        return self.val * other

    def __rtruediv__(self, other):
        return self.val / other

    def __pow__(self, power, modulo=None):
        return pow(self.val, power, modulo)


class LinAnneal(Anneal):
    def __init__(self, params=None):
        self.initialize_params(params)

        self.val = self.start_value

    def anneal(self, current_step):
        ratio = max((self.number_of_steps - current_step) /
                    float(self.number_of_steps), 0)
        self.val = (self.start_value - self.end_value) * ratio + self.end_value

    def initialize_params(self, params):
        super().initialize_params(params)

        self.start_value = self.params["start_value"]
        self.end_value = self.params["end_value"]
        self.number_of_steps = self.params["num_steps"]
        self.anneal_type = self.params["anneal_type"] = "lin"


class ExpAnneal(Anneal):
    def __init__(self, params=None):
        self.initialize_params(params)

        self.val = self.start_value

    def anneal(self, current_step):
        """
        Implementation adapted from https://stackoverflow.com/a/50903555/5418798
        """
        start = np.log(self.start_value)
        stop = np.log(self.end_value)
        incr = (stop - start) / self.number_of_steps
        x = start + current_step * incr
        new_value = np.exp(x)
        if self.start_value > self.end_value:
            self.val = max(self.end_value, new_value)
        else:
            self.val = min(self.end_value, new_value)

    def initialize_params(self, params=None):
        super().initialize_params(params)

        self.start_value = self.params["start_value"]
        self.end_value = self.params["end_value"]
        self.number_of_steps = self.params["num_steps",
                                           "The number of steps to go from start to end value"]
        self.anneal_type = self.params['store']["anneal_type"] = "exp"
