# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod
from diadem.common import BaseObject


class Environment(BaseObject):

    def __init__(self, params=None):
        self.initialize()
        self._initialize_params(params=params)

    def _initialize_params(self, params):
        super()._initialize_params(params=params)
        self.__action_timedelta = self.params["action_timedelta"]
        
    @abstractmethod
    def step(self, actions):
        return  # next_observation, reward, done, info

    def reset(self, state=None):
        return  # state, init_error

    @abstractmethod
    def render(self, filename=None, show=True):
        pass

    @abstractmethod
    def create_init_states(self, size=None, idx=None):
        """ creates a list of initial environment states which can be used in reset to set the state of the environment
        
        Keyword Arguments:
            idx {[indices list] or single index} -- [description] (default: {None})
            size {[int]} -- [total size of the data set] (default: {None})
        Returns:
            a single state if idx is a single index or list of states if idx is an indices list
        """
        pass

    @property
    @abstractmethod
    def actionspace(self):
        pass

    @property
    @abstractmethod
    def observationspace(self):
        pass

    @property
    @abstractmethod
    def contains_training_data(self):
        return False

    @abstractmethod
    def state_as_string(self):
        pass

    @property
    @abstractmethod
    def action_timedelta(self):
        return self.__action_timedelta

    @action_timedelta.setter
    def action_timedelta(self, action_timedelta):
        self.__action_timedelta = action_timedelta
