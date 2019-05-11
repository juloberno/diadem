# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import os
from diadem.experiment.visualizers import BaseVisualizer

class DumpImageVisualizer(BaseVisualizer):
    def __init__(self, params, context, *args, **kwargs):
        super().__init__(params, context, *args, **kwargs)

        self.realtime_factor = params['realtime_factor']

    def _render(self, *args, **kwargs):
        figures_path = self.params['figures_path'].format(**kwargs)
        figure_filename = self.params['figure_filename'].format(**kwargs)
        img_path = os.path.join(figures_path, figure_filename)

        os.makedirs(figures_path, exist_ok=True)
        
        self.context.environment.render(filename=img_path, show=False)

        self.context.summary_service.static_fields["environment_screenshot"] = img_path

class DumpFirstAndLastImageVisualizer(DumpImageVisualizer):
    def _render(self, *args, **kwargs):
        if kwargs.get('step') != 1 and not kwargs.get('last_step'):
            return
        
        super()._render(*args, **kwargs)