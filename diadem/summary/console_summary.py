# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.summary.pandas_summary import PandasSummary

class ConsoleSummary(PandasSummary):
    def __init__(self, summary_cols=None):
        super().__init__(summary_cols=summary_cols)

    def dump(self, path, reset=True):
        print(self.df)
    
    def load(self, path):
        pass
