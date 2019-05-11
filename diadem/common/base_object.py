# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod
import os
from diadem.common.params import Params
import logging


class BaseObject(object):
    def __init__(self):
        pass

    def initialize(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def write_parameters(self, filename=None, print_descriptions=False):
        n = type(self).__name__
        fn = "Params" + n + ".json"
        if filename is not None:
            fn = filename

        self.params.save(filename=fn, print_description=print_descriptions)

    def _initialize_params(self, params=None):
        if params is None:
            self.params = Params()
        else:
            self.params = params
