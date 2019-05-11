# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import time
from diadem.experiment.visualizers import BaseVisualizer

class OnlineVisualizer(BaseVisualizer):
    def __init__(self, params, context, *args, **kwargs):
        super(OnlineVisualizer, self).__init__(params, context, *args, **kwargs)

        self.realtime_factor = params['realtime_factor']

    def _render(self, *args, **kwargs):
        super(OnlineVisualizer, self)._render(*args, **kwargs)
        
        time.sleep(self.context.environment.action_timedelta /
                    self.realtime_factor)