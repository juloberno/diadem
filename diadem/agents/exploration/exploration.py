# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod
from diadem.common import BaseObject


class Exploration(BaseObject):
    def __init__(self, params):
        self._initialize_params(params)

    @abstractmethod
    def explore(self):
        return False

    def restore(self, directory):
        pass

    def freeze(self, directory):
        pass

    def anneal(self, current_step):
        pass
