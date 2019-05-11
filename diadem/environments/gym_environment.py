# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import gym
import pyglet
from .environment import Environment

class GymEnvironment(Environment):

    def __init__(self, gym_name='CartPole-v0', params=None):
        super().__init__()
        self.gym_name = gym_name
        self.env = gym.make(gym_name)

    def _initialize_params(self, params=None):
        super()._initialize_params(params=params)

    def step(self, actions):
        next_observation, reward, done, info = self.env.step(actions)
        return next_observation, reward, done, info

    def reset(self, state=None):
        current_observation = self.env.reset()
        return current_observation, False

    def create_init_states(self, size=None, idx=None):
        pass

    def render(self, filename=None, show=True):
        try:
            self.env.render()
            if filename is not None:
                pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
        except:
            self.log.warning("Vide screen not available in GymEnvironment")


    @property
    def actionspace(self):
        return self.env.action_space

    @property
    def observationspace(self):
        return self.env.observation_space

    def contains_training_data(self):
        return True

    def state_as_string(self):
        return self.gym_name

    @property
    def action_timedelta(self):
        # only for visualization purposes actual step time unknown
        return 0.1

