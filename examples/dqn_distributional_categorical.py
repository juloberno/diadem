# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


"""
Run the experiment

Important sidemark: the Agent is defined in the parameters, not in the main file!


"""

from examples.dqn_basic import run_dqn_algorithm

if __name__ == '__main__':
    # basic Double DQN with Prioritized Experience Replay
    run_dqn_algorithm(parameter_files=["examples/example_params/common_parameters.yaml",
                                                 "examples/example_params/dqn_basic.yaml",
                                                 "examples/example_params/dqn_distributional_categorical.yaml"])