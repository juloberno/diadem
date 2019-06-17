# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import os

from diadem.environments import Environment
from bark_project.modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution import UniformVehicleDistribution
from bark_project.modules.runtime.ml.runtime_rl import RuntimeRL
from bark_project.modules.runtime.ml.nn_state_observer import StateConcatenation
from bark_project.modules.runtime.ml.action_wrapper import MotionPrimitives
from bark_project.modules.runtime.ml.state_evaluator import GoalReached
from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_project.modules.runtime.viewer.pygame_viewer import PygameViewer

class BarkHighway(Environment):
    def __init__(self, params=None):
        super().__init__()
        os.chdir(os.path.join(os.getcwd(), "external/bark_project/"))
        params = ParameterServer(filename="modules/runtime/tests/data/highway_merging_with_rl_and_vis.json")
        scenario_generation = UniformVehicleDistribution(num_scenarios=3, random_seed=0, params=params)
        state_observer = StateConcatenation(params=params)
        action_wrapper = MotionPrimitives(params=params)
        evaluator = GoalReached(params=params)
        viewer = PygameViewer(params=params, x_range=[-40,40], y_range=[-40,40], follow_agent_id=True) #use_world_bounds=True) # 

        self.runtime = RuntimeRL(action_wrapper=action_wrapper, nn_observer=state_observer,
                        evaluator=evaluator, step_time=0.2, viewer=viewer,
                        scenario_generator=scenario_generation)

    def _initialize_params(self, params=None):
        super()._initialize_params(params=params)

    def step(self, actions):
        next_observation, reward, done, info = self.runtime.step(actions)
        return next_observation, reward, done, info

    def reset(self, state=None):
        current_observation = self.runtime.reset()
        return current_observation, False

    def create_init_states(self, size=None, idx=None):
        pass

    def render(self, filename=None, show=True):
        self.runtime.render()

    @property
    def actionspace(self):
        return self.runtime.action_space

    @property
    def observationspace(self):
        return self.runtime.observation_space

    def contains_training_data(self):
        return True

    def state_as_string(self):
        return "BarkEnvironment"

    @property
    def action_timedelta(self):
        # only for visualization purposes actual step time unknown
        return self.runtime.step_time
