# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import pandas as pd
import os
from diadem.summary.base_summary import BaseSummary
import logging

class PandasSummary(BaseSummary):
    df = pd.DataFrame()

    def reset(self):
        self.df.drop(self.df.index, inplace=True)

    def add(self, summary_row):
        def _format_row(row):
            if self.summary_cols is None:
                return row
            
            return {key: value for key, value in row.items() if key in self.summary_cols}

        if isinstance(summary_row, list):
            rows = [_format_row({**self.static_fields, **row}) for row in summary_row]
        else:
            rows = [_format_row({**self.static_fields, **summary_row})]
        self.df = self.df.append(rows, ignore_index=True)

    def dump(self, path, reset=True):
        if not self.df.empty:
            is_first_dump = not os.path.exists(path)
            self.df.to_csv(path, mode='a',
                           index=False, header=is_first_dump)
            log_string = self.df.tail(10).mean()
            if reset:
                self.reset()
            return log_string

    def load(self, path):
        if os.path.exists(path):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.DataFrame()
