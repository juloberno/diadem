# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


"""
Run the experiment with Video Rendering

Important sidemark: the Agent is defined in the parameters, not in the main file!

Example: 

  NUM_PARTICIPANTS=3 MAIN_DIR=../experiment python3 main.py

"""

import tensorflow as tf

from diadem.agents import AgentContext, AgentManager
from diadem.environments import GymEnvironment
from diadem.experiment import Experiment
from diadem.experiment.visualizers import VideoVisualizer
from diadem.preprocessors import Normalization
from diadem.summary import ConsoleSummary
from diadem.common import Params


def main():
    exp_dir = "tmp_exp_dir"
    params = Params(filename=["examples/example_params/common_parameters.yaml",
                                                 "examples/example_params/video_rendering.yaml"])
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
                         visualizer=VideoVisualizer(params=params['visualization'], context=context))
    exp.run()


if __name__ == '__main__':
    main()