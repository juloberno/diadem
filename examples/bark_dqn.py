# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

import tensorflow as tf
from diadem.environments import BarkHighway
from diadem.agents import AgentContext, AgentManager
from diadem.experiment import Experiment
from diadem.experiment.visualizers import OnlineVisualizer
from diadem.summary import PandasSummary
from diadem.common import Params
from diadem.preprocessors import Normalization

def run_dqn_algorithm(parameter_files):
    exp_dir = "tmp_exp_dir"
    params = Params(filename=parameter_files)
    environment = BarkHighway(params=params['environment'])
    context = AgentContext(
        environment=environment,
        datamanager=None,
        preprocessor=Normalization(environment=environment),
        optimizer=tf.train.AdamOptimizer,
        summary_service=PandasSummary()
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
                                                 "examples/example_params/dqn_basic.yaml",
                                                 "examples/example_params/dqn_distributional_categorical.yaml"])