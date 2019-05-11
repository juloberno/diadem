# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import abstractmethod
import tensorflow as tf
from diadem.common import BaseObject


class Preprocessing:
    def __init__(self):
        self.preprocessor_list = []

    def add(self, preprocessor):
        self.preprocessor_list.append(preprocessor)

    def preprocess(self, observation):
        for preprocessor in self.preprocessor_list:
            observation = preprocessor.preprocess(observation)

        return observation

    def preprocessed_shape(self, shape):
        for preprocessor in self.preprocessor_list:
            shape = preprocessor.preprocessed_shape(shape)

        return shape


class Preprocessor(BaseObject):
    def __init__(self, variable_scope='preprocessor', params=None):
        super(Preprocessor, self)._initialize_params(params=params)
        super(Preprocessor, self).initialize()

    @abstractmethod
    def preprocessed_shape(self, shape):
        return shape

    @abstractmethod
    def preprocess(self, observation):
        return observation
