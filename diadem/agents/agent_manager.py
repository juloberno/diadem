# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import os

from diadem.agents.model import QNetwork, DistributionalQNetwork
from diadem.environments import Environment

from diadem.agents import Agent, DQfDAgent, DistributionalDQfDAgent, HbaAgent, EnsembleAgent


class AgentManager(Agent):
    _agent_classes = {
        'hba': HbaAgent,
        'dqfd': DQfDAgent,
        'distr_dqfd': DistributionalDQfDAgent,
        'ensemble': EnsembleAgent
    }
    _network_classes = {
        'q_fully': QNetwork,
        'dq_fully': DistributionalQNetwork
    }
    agents = {}
    main_agent = None

    def __init__(self, params=None, context = None):
        super().__init__(params=params, context=context)
        context.agent_manager = self
        self.context = context

        for name, agent_params in self.params['agents'].items():
            num_instances = agent_params['num_instances']
            for instance_id in range(0,num_instances):
                agent_type = str(agent_params['agent_type'])
                network_type = agent_params['network']['type']
                network = self._agent_network(agent_type, network_type)

                if num_instances > 1:
                    agent_name = "{}{}".format(name,instance_id)
                else:
                    agent_name = name

                self.agents[agent_name] = self._agent_classes[agent_type](
                    params=agent_params,
                    context=self.context.clone(agent_params),
                    network_model=network
                )
                self.agents[agent_name].name = agent_params['name'] or agent_name
                if agent_params['is_main_agent']:
                    if num_instances == 1:
                        self.main_agent = self.agents[agent_name]
                    else:
                        raise ValueError("There can only exist a single main agent. \
                                    You passed num_istances = {}".format(num_instances))


    def _agent_network(self, agent_type, network_type):
        network_class = None
        if agent_type == 'dqfd':
            if network_type == 'fully_connected':
                network_class = QNetwork
            else:
                raise ValueError('Undefined network type')
        if agent_type == 'distr_dqfd':
            if network_type == 'fully_connected':
                network_class = DistributionalQNetwork
            else:
                raise ValueError('Undefined network type')
        return network_class

    def initialize(self, *args, **kwargs):
        """
        Initialize all agents

        Initializes and sets the context for every agent
        """
        for name, agent in self.agents.items():
            agent_kwargs = {
                **kwargs,
                'evaluations_directory': os.path.join(kwargs.get('evaluations_directory', './'), name),
                'log_directory': os.path.join(kwargs.get('log_directory', './'), name),
                'checkpoints_directory': kwargs.get('checkpoints_directory_' + name, os.path.join(kwargs.get('checkpoints_directory', './'), name))
            }
            agent.initialize(*args, **agent_kwargs)

    def initialize_summary_writer(self, *args, **kwargs):
        for agent in self.agents.values():
            agent.initialize_summary_writer(*args, **kwargs)

    def observe(self, *args, **kwargs):
        for agent in self.agents.values():
            if agent.observe_via_agent_manager:
                agent.observe(*args, **kwargs)
            
    def close(self):
        for agent in self.agents.values():
            agent.close()

    def get_next_best_action(self, *args, **kwargs):
        return self.main_agent.get_next_best_action(*args, **kwargs)

    def agent(self, name):
        """
        Get an agent instance by its name

        Parameters
        ----------
        name : str

        Raises
        ------
        KeyError
            If no agent with the name is registered

        Returns
        -------
        Agent
        """
        return self.agents[name]

    def reset(self):
        for agent in self.agents.values():
            agent.reset()

    def freeze(self, *args, **kwargs):
        for agent in self.agents.values():
            agent.freeze(*args, **kwargs)

    def restore(self, *args, **kwargs):
        for agent in self.agents.values():
            agent.restore(*args, **kwargs)

    def update_model(self, *args, **kwargs):
        for agent in self.agents.values():
            agent.update_model(*args, **kwargs)

    def param_annealing(self, *args, **kwargs):
        for agent in self.agents.values():
            agent.param_annealing(*args, **kwargs)

    @property
    def explorer(self):
        return self.main_agent.explorer

    @property
    def per_beta(self):
        return self.main_agent.per_beta
