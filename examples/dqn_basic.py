# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


"""
Run the experiment

Important sidemark: the Agent is defined in the parameters, not in the main file!


"""

import tensorflow as tf

from diadem.agents import AgentContext, AgentManager
from diadem.environments import GymEnvironment
from diadem.experiment import Experiment
from diadem.experiment.visualizers import OnlineVisualizer
from diadem.summary import ConsoleSummary
from diadem.common import Params
from diadem.preprocessors import Normalization


def run_dqn_algorithm(parameter_files):
    exp_dir = "tmp_exp_dir"
    params = Params(filename=parameter_files)
    environment = GymEnvironment(params=params['environment'])
    context = AgentContext(
        environment=environment,
        datamanager=None,
        preprocessor=Normalization(environment=environment),
        optimizer=tf.train.AdamOptimizer,
        summary_service=ConsoleSummary()
    )
    agent = AgentManager(
        params=params,
        context=context
    )

    exp = Experiment(
        params=params['experiment'], main_dir=exp_dir, context=context, agent=agent,
                         visualizer=None)
    exp.run()




if __name__ == '__main__':
    # basic Double DQN with Prioritized Experience Replay
    run_dqn_algorithm(parameter_files=["examples/example_params/common_parameters.yaml",
                                                 "examples/example_params/dqn_basic.yaml"])