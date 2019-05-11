# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.common import BaseObject

class BaseSummary(BaseObject):
    def __init__(self, summary_cols=None):
        self.static_fields = {}
        self.summary_cols = summary_cols
        self.reset()

    def add(self, summary_row):
        raise NotImplementedError()

    def dump(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
