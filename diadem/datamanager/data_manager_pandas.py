# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.datamanager.data_manager import DataManager
import pandas as pd
import numpy as np
from os import path
import json


class DataManagerPandas(DataManager):

    def read_data_set(self, data_set_name):
        df = pd.read_pickle(path.join(self.data_folder, data_set_name))
        samples = np.array([{
            **dict(row),
            'states': json.loads(row['states'])
        } for _, row in df.iterrows()])

        return samples

    def dump_data_set(self, data_set_name, samples):
        samples = [{
            **sample,
            'states': json.dumps(np.array(sample['states']).tolist())
        } for sample in samples]
        df = pd.DataFrame(data=samples)
        df.to_pickle(path.join(self.data_folder, data_set_name))