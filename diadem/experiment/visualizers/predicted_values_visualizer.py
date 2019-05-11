# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.experiment.visualizers import OnlineVisualizer, BaseVisualizer, VideoRenderer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time, json, os
from collections import deque
matplotlib.use("Agg")

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

import matplotlib.backends.backend_agg as agg

import pylab

import pygame, shutil
from pygame.locals import *


class QValuesPlotsVisualizer(BaseVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agents = ['passive_agent', 'aggressive_agent']
        self.num_actions = self.context.agent_manager.main_agent.num_discrete_actions
        self.history = []

        self.plot_q = kwargs.get('plot_q')
        self.plot_q_bar = kwargs.get('plot_q_bar')
        self.plot_p = not self.plot_q and kwargs.get('plot_p')

        if not self.plot_q and not self.plot_p and not self.plot_q_bar:
            raise ValueError('Please set either plot_p or plot_q')

        self.figs = []
        self.axes = []
        for _ in range(len(self.agents)+1):
            fig = pylab.figure(figsize=[4, 4], dpi=100)
            ax = fig.gca()
            self.figs.append(fig)
            self.axes.append(ax)

        if self.plot_p:
            self.limits = None

            nb_atoms = self.context.agent_manager.agent([name for name, agent in self.context.agent_manager.agents.items() if name != 'hba_agent'][0]).params['network']['num_atoms']
            tau = np.arange(0, nb_atoms + 1) / nb_atoms

            # extend tau with [0, tau, 1]
            self.tau = np.zeros(nb_atoms+2)
            self.tau[1:-1] = (tau[1:] + tau[:-1]) / 2
            self.tau[-1] = 1.
            
            self.plots = [[self.axes[agent].step(np.arange(0, nb_atoms+2), self.tau)[0] for _ in range(self.num_actions)] for agent in range(len(self.agents))]


    def _render(self, *args, **kwargs):
        if len(self.context.summary_service.df) == 0:
            return

        df = self.context.summary_service.df

        def parse(x):
            if type(x) is str:
                return json.loads(x)
            else:
                return x
        agents = self.agents + ['hba']

        if self.plot_q_bar:
            values = [[list(self.context.summary_service.df['q_value_{}_{}'.format(agent, action)])[-1] for action in range(self.num_actions)] for agent in agents]
        if self.plot_q:
            self.history.append([[list(self.context.summary_service.df['q_value_{}_{}'.format(agent, action)])[-1] for action in range(self.num_actions)] for agent in agents])
        if self.plot_p:
            self.history.append([[parse(list(self.context.summary_service.df['p_values_{}_{}'.format(agent, action)])[-1]) for action in range(self.num_actions)] for agent in agents])
        
        self.context.environment.render(show=False)
        x_pos = [10, 590, 350]
        y_pos = [10, 10, 600]
        
        for agent, name in enumerate(agents):
            if name != 'hba':
                posterior_col = 'posterior_percent_{}'.format(name)
                posterior = int(list(self.context.summary_service.df[posterior_col])[-1]*100)
            if self.plot_q:
                values = [hist_item[agent] for hist_item in self.history]
                self.axes[agent].plot(values)
            if self.plot_q_bar:
                if name != 'hba':
                    self.axes[agent].set_title('Q Values of {} ({} \%)'.format(name.replace('_', ' '), posterior))
                else:
                    self.axes[agent].set_title('Q Values of HBA')
                q_values = [x + abs(min(values[agent])) for x in values[agent]]
                relative_q_values = [x / sum(q_values) for x in q_values]
                bars = self.axes[agent].bar(('-3 $\\frac{m}{s^2}$', '0 $\\frac{m}{s^2}$', '2 $\\frac{m}{s^2}$', '5 $\\frac{m}{s^2}$'), relative_q_values, color=['#4DCB4A' if x == max(values[agent]) else '#4D8F4A' for x in values[agent]], label="Example {}".format(kwargs['step']))
                # self.axes[agent].set_yticklabels([])
            if self.plot_p:
                quant_out = self.history[-1][agent]
                if self.limits is None:
                    self.limits = [np.min(quant_out), np.max(quant_out)]
                else:
                    self.limits = [min(np.min(quant_out), self.limits[0]), max(np.max(quant_out), self.limits[1])]

                self.limits[0] = max(self.limits[0], -1500)
                self.axes[agent].set_xlim(self.limits)
                
                self.axes[agent].set_title('Distributional Function of {} ({} \%)'.format(self.agents[agent].replace('_', ' '), posterior))
                self.axes[agent].legend(['-3 $\\frac{m}{s^2}$', '0 $\\frac{m}{s^2}$', '2 $\\frac{m}{s^2}$', '5 $\\frac{m}{s^2}$'], loc='upper left')
                for line, quant in zip(self.plots[agent], quant_out):
                    x_data = np.zeros(len(quant)+2)
                    x_data[1:-1] = quant
                    x_data[0] = self.limits[0]
                    x_data[-1] = self.limits[-1]
                    line.set_xdata(x_data)

            canvas = agg.FigureCanvasAgg(self.figs[agent])
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            size = canvas.get_width_height()
            screen = pygame.display.get_surface()
            image = pygame.image.fromstring(raw_data, size, "RGB")
            screen.blit(image, (x_pos[agent], y_pos[agent]))
            plt.close('all')

            if self.plot_q_bar:
                bars.remove()

        if kwargs.get('last_step'):
            self.history.clear()

class OnlineVisualizerQValuesPlots(QValuesPlotsVisualizer):
    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)

        pygame.display.update()

        time.sleep(self.context.environment.action_timedelta /
                        self.realtime_factor)

class VideoQValuesVisualizer(QValuesPlotsVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params['render_every_n_steps'] = 1
        self.params['figure_filename'] = "{step}.png"
        self._framerate = self.params['framerate'] or (1 / 0.2)
    
    def _render(self, *args, **kwargs):
        if kwargs.get('step') == 1:
            figures_path = os.path.abspath(self.params['figures_path'].format(**kwargs))
            if os.path.exists(figures_path):
                shutil.rmtree(figures_path)

        super()._render(*args, **kwargs)
        figures_path = self.params['figures_path'].format(**kwargs)
        figure_filename = self.params['figure_filename'].format(**kwargs)
        img_path = os.path.join(figures_path, figure_filename)

        os.makedirs(figures_path, exist_ok=True)
        
        pygame.image.save(pygame.display.get_surface(), img_path)

        self.context.summary_service.static_fields["environment_screenshot"] = img_path

        if kwargs.get('last_step'):
            figures_path = os.path.abspath(self.params['figures_path'].format(**kwargs))
            video_dir = os.path.abspath(self.params['video_path'].format(**kwargs))
            self.log.info('Create video for %s and %s' % (figures_path, video_dir))
            _video_renderer = VideoRenderer(fig_path=figures_path, video_name='./', clear_figures=False)
            _video_renderer.export_video(filename=video_dir, framerate=self._framerate, remove_video_dir=False)