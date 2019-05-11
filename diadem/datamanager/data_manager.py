# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod
import os
from os.path import join
import numpy as np
import random
from diadem.common import BaseObject
import pickle


class DataManager(BaseObject):
    def __init__(self, preprocessor=None, params=None, base_dir=None):
        self._initialize_params(params=params, base_dir=base_dir)
        self.preprocessor = preprocessor

        onlyfiles = [f for f in os.listdir(
            self.data_folder) if os.path.isfile(join(self.data_folder, f))]
        self.test_data_files = []
        self.train_data_files = []
        self.critical_data_files = []
        self.current_test_idx = -100
        self.current_train_idx = -100
        self.current_test_batch_idx = []
        self.current_train_batch_idx = []

        for file in onlyfiles:
            if "train" in file:
                self.train_data_files.append(join(self.data_folder, file))
            elif "test" in file:
                self.test_data_files.append(join(self.data_folder, file))
            elif "evaluate" in file:
                self.train_data_files.append(join(self.data_folder, file))

        self.train_data = []
        self.test_data = []

    def _initialize_params(self, params=None, base_dir=None):
        super()._initialize_params(params=params)

        self.base_dir = base_dir if base_dir is not None else ''
        self.data_folder = join(
            self.base_dir, params["data_folder", "folder where the data lies."])

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def get_data_set(self, files, findstr):
        templist = []
        for file in files:
            if findstr in file:
                templist.append(file)

        if len(templist) > 1:
            self.log.info(
                "more than one data file containing this substring found. Choosing first one.")
        elif templist is None:
            raise ValueError(
                "could not find any data containing this substring")
        data_set_name = templist[0]
        data = self.read_data_set(data_set_name=data_set_name)

        return data

    @abstractmethod
    def read_data_set(self, data_set_name):
        pass

    @abstractmethod
    def dump_data_set(self, samples):
        pass

    def set_train_data(self, findstr):
        self.train_data = self.get_data_set(self.train_data_files, findstr)

    def set_test_data(self, findstr):
        self.test_data = self.get_data_set(self.test_data_files, findstr)

    def get_next_test_sample(self):
        if self.current_test_idx < 0 or self.current_test_idx > self.test_data.shape[0]:
            self.current_test_idx = 0
            self.log.info("reinit train data index to zero")
        data = self.test_data[self.current_test_idx]
        self.current_test_idx += 1
        return data

    def get_idx_test_sample(self, idx):
        if self.current_test_idx < 0 or self.current_test_idx > self.train_data.shape[0]:
            self.current_test_idx = 0
            self.log.info("reinit test data index to zero")
        self.current_test_idx = idx
        return self.test_data[self.current_test_idx]

    def get_random_test_sample(self):
        idx = random.randrange(0, self.test_data.shape[0])
        self.current_test_idx = idx
        return self.test_data[idx]

    def get_next_train_sample(self):
        if self.current_train_idx < 0 or self.current_train_idx > self.train_data.shape[0]:
            self.current_train_idx = 0
            self.log.info("reinit train data index to zero")
        data = self.train_data[self.current_train_idx]
        self.current_train_idx += 1
        return data

    def get_idx_train_sample(self, idx):
        if self.current_train_idx < 0 or self.current_train_idx > self.train_data.shape[0]:
            self.current_train_idx = 0
            self.log.info("reinit train data index to zero")
        self.current_train_idx = idx
        return self.train_data[self.current_train_idx]

    def get_random_train_sample(self):
        idx = random.randrange(0, self.train_data.shape[0])
        self.current_train_idx = idx
        return self.train_data[idx]

    def get_random_train_batch(self, batch_size):
        idx = random.sample(range(self.train_data.shape[0]), batch_size)
        self.current_train_batch_idx = idx
        return self.train_data[idx]

    def get_random_test_batch(self, batch_size):
        idx = random.sample(range(self.test_data.shape[0]), batch_size)
        self.current_test_batch_idx = idx
        return self.test_data[idx]
