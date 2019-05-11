# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.agents.exploration import Exploration
import numpy as np


class Epsilon(Exploration):
    def __init__(self, params):
        super().__init__(params)
        self.initialize()

    def _initialize_params(self, params=None):
        super()._initialize_params(params)

        self.epsilon = self.params["epsilon",
                                   "probability threshold determining if random action is taken", "exp"]

    def explore(self):
        if np.random.random() > self.epsilon:
            return False
        else:
            return True

    def anneal(self, current_step):
        self.epsilon.anneal(current_step=current_step)
