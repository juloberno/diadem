# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.common import BaseObject

class BaseVisualizer(BaseObject):
    def __init__(self, params, context, *args, **kwargs):
        self.params = params
        self.context = context
        self.initialize()

    def visualize(self, *args, **kwargs):
        render_step = self.params['render_every_n_steps'] and self.params['render_every_n_steps'] > 0 and kwargs.get('step') % self.params['render_every_n_steps'] == 0
        render_episode = self.params['render_every_n_episodes'] and self.params['render_every_n_episodes'] > 0 and kwargs.get('episode') % self.params['render_every_n_episodes'] == 0
        
        if render_step and render_episode:
            self._render(*args, **kwargs)

    def _render(self, *args, **kwargs):
        self.context.environment.render()