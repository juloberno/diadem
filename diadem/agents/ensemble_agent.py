# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.agents.agent import Agent
from collections import deque
import numpy as np
import json
from scipy.stats import norm

class EnsembleAgent(Agent):
    def __init__(self, context = None, params=None, *args, **kwargs):
        super().__init__(context=context, params=params)
        self.name = 'ensemble_agent'

        # this variable defines which sub agent is responsible for action selection in current episode
        self.policy_sub_agent = None   

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.sub_agents =  [sub_agent_name for sub_agent_name, _ in self.context.agent_manager.agents.items() \
                                    if self.agents_name_prefix in sub_agent_name.lower()] 
        self.num_sub_agents = len(self.sub_agents)
        if self.num_sub_agents < 1:
            raise ValueError("Ensemble agents could not find any subagents.")

        self._reset_policy_sub_agent()

    def _initialize_params(self, params=None):
        super()._initialize_params(params)
        self.agents_name_prefix = self.params["agents_name_prefix"]

    def observe(self, observation, action, reward, next_observation, done, info={}, guided=False):
        super().observe(observation=observation, action=action, reward=reward,
                        next_observation=next_observation, done=done, guided=guided)

        if done:
            self._reset_policy_sub_agent()

        for sub_agent in self.sub_agents:
            smp = np.random.uniform(0,1)
            # let sub agents observe only with 50% probability -> make maybe parameterizable
            if smp < 0.5:
                self.context.agent_manager.agent(sub_agent).observe(observation, action, reward, \
                                                             next_observation, done, info, guided)
    
    def _reset_policy_sub_agent(self):
        sub_agent_id = np.random.randint(0, self.num_sub_agents)
        self.policy_sub_agent = self.sub_agents[sub_agent_id]
        self._add_to_summary("policy_sub_agent", self.policy_sub_agent )


    def get_next_best_action(self, observation, *args, **kwargs):
        if self.policy_sub_agent:
            return self.context.agent_manager.agent(self.policy_sub_agent).get_next_best_action(observation, *args, **kwargs)
        else:
<<<<<<< HEAD
            raise ValueError("No policy sub agent available")
=======
            raise ValueError("No policy sub agent available")



     

>>>>>>> origin/master
