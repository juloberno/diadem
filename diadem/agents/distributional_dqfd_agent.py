# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from collections import deque
import numpy as np
import random
import os
import tensorflow as tf
import tempfile
import time
import json
import pickle

from diadem.agents.model import DistributionalQNetwork
from diadem.agents import DQfDAgent
from diadem.agents.model import QNetwork
from diadem.agents.losses import quantile_loss, categorical_loss


class DistributionalDQfDAgent(DQfDAgent):
    def _calculate_loss(self):
        loss_func = quantile_loss if self.params["network"]["quantile_regression"] else categorical_loss

        with tf.name_scope('td_loss'):
            td_errors = loss_func(
                target_network=self.target_q_network,
                estimation_network=self.action_prediction_network,
                actions=self.actions,
                evaluation_actions=self.action_selection,
                rewards=self.rewards,
                done_mask=self.done_mask,
                params={**self.params['network'], **self.params['loss']}
            )
            td_loss = tf.reduce_mean(td_errors, name='td_loss')

        n_step_td_loss = None
        n_step_errors = None
        if self.n_step > 1:
            with tf.name_scope('td_nstep_loss'):
                n_step_td_errors = loss_func(
                    target_network=self.target_n_step_q_network,
                    estimation_network=self.n_step_prediction_network,
                    actions=self.actions,
                    evaluation_actions=self.action_selection_n_step,
                    rewards=self.n_step_rewards,
                    done_mask=self.done_mask,
                    params={**self.params['network'], **self.params['loss']}
                )
                n_step_td_loss = tf.reduce_mean(
                    n_step_td_errors, name='n_step_td_loss')

        return td_errors, td_loss, n_step_errors, n_step_td_loss

    def p_values(self, observation):
        observation = self.env_state_to_nn_input(observation)

        tf.reset_default_graph()
        with self.context.graph.as_default():
            return self.context.session.run(
                self.action_prediction_network.p,
                self.fill_session_dict(
                    self.states_input, observation)
            )[0]
