# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import shutil
import os

from diadem.experiment.evaluation.export import VideoRenderer
from diadem.experiment.visualizers import DumpImageVisualizer



class VideoVisualizer(DumpImageVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params['render_every_n_steps'] = 1
        self.params['figure_filename'] = "{step}.png"
        self._framerate = self.params['framerate']
    
    def _render(self, *args, **kwargs):
        if kwargs.get('step') == 1:
            figures_path = os.path.abspath(self.params['figures_path'].format(**kwargs))
            if os.path.exists(figures_path):
                shutil.rmtree(figures_path)

        super()._render(*args, **kwargs)
        
        if kwargs.get('last_step'):
            figures_path = os.path.abspath(self.params['figures_path'].format(**kwargs))
            video_dir = os.path.abspath(self.params['video_path'].format(**kwargs))
            self.log.info('Create video for %s and %s' % (figures_path, video_dir))
            _video_renderer = VideoRenderer(fig_path=figures_path, video_name='./', clear_figures=False)
            _video_renderer.export_video(filename=video_dir, framerate=self._framerate,remove_video_dir=False)
 