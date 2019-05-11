# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from .data_manager import DataManager
import pandas as pd
import numpy as np
from os import path


class DataManagerCSV(DataManager):

    def read_data_set(self, data_set_name):
        data = pd.read_csv(data_set_name)
        return np.array(data)

    def dump_data_set(self, data_set_name, samples):
        df = pd.DataFrame(samples)
        df.to_csv(path.join(self.data_folder, data_set_name))
