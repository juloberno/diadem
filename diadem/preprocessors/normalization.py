# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
import numpy as np

from diadem.preprocessors.preprocessor import Preprocessor



class Normalization(Preprocessor):
    """
        Normalize observation to range 0 to 1
    """
    def __init__(self, environment, variable_scope='normalization', params=None):
        super(Normalization, self).__init__(variable_scope=variable_scope, params=params)
        self.environment = environment

    def preprocess(self, observation):
        return (observation - self.environment.observationspace.low)/ (self.environment.observationspace.high-self.environment.observationspace.low)

    def preprocessed_shape(self, shape):
        return shape


