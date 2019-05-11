# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.datamanager.data_manager import DataManager
import pandas as pd
import numpy as np
from os import path


class DataManagerNumPy(DataManager):

    def read_data_set(self, data_set_name):
        return np.load(path.join(self.data_folder, data_set_name))

    def dump_data_set(self, data_set_name, samples):
        np.save(path.join(self.data_folder, data_set_name), np.array(samples))