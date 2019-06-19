# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
import os

from collections import deque
import numpy as np
import random
import time
import struct
import json

from diadem.agents.experience_replay_agent import ExperienceReplayAgent
from diadem.agents.model import QNetwork
from diadem.agents.exploration import Epsilon
from diadem.agents.agent_context import AgentContext

from gym import spaces

def namescope(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.name_scope(name):
                func(*args, **kwargs)
        return wrapper
    return decorator


class DQfDAgent(ExperienceReplayAgent):
    def __init__(self, context: AgentContext, params, network_model: QNetwork, *args, **kwargs):
        super().__init__(context=context, params=params, *args, **kwargs)
        self._network_model = network_model
        self.name = 'DQfDAgent'

    def _initialize_params(self, params=None):
        super()._initialize_params(params=params)

        self._summary_debug_enabled = self.params['summary']['debug']

        # Q learning parameters
        self.supervised_margin = self.params["supervised_margin"]

        self.exploration_type = self.params["exploration"]["type",
                                                           "Determines the type of exploration: random, epsilon"]
        if self.exploration_type == "random":
            self.exploration_rate = self.params["exploration"]
            self.explorer = Epsilon(params=self.exploration_rate)

        self.initialize_guidance_steps = self.params["exploration"]["initialize_guidance_steps",
                                                                    "how many steps are we doing pure guidance at the beginning of training."]
        self.hastar_guiding_factor = self.params["guidance"]["guiding_factor"]

        self.target_update_rate = self.params["target_update_rate"]
        self.double_q_learning = self.params["double_q_learning"]
        
        if isinstance(self.context.environment.actionspace, spaces.Discrete):
            self.num_discrete_actions = self.context.environment.actionspace.n
        else:
            raise ValueError("DQ(X) Agent only supports discrete action spaces.")

    def create_variables(self):
        super().create_variables()

        self.input()
        self.predict_actions()
        self.estimate_future_rewards()
        self.compute_temporal_differences()
        self.update_target_network()
        self.create_summaries()

    def _q_network(self, name, states, *args, **kwargs):
        input_dim = self.get_nn_inputs_definition()[
            "env_state"]
        layer_in = states["env_state"]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return self._network_model(
                layer_in=layer_in,
                dim_in=input_dim,
                dim_out=self.num_discrete_actions,
                params=self.params['network'],
                *args, **kwargs
            )

    def target_network(self, states, *args, **kwargs):
        return self._q_network('target_network', states, *args, **kwargs)+self.prior_network

    def prediction_network(self, states, *args, **kwargs):
        return self._q_network('prediction_network', states, *args, **kwargs)+self.prior_network

    @namescope('input')
    def input(self):
        self.done_mask = tf.placeholder(tf.float32, [None], name="done")
        self.action_mask = tf.placeholder(
            tf.float32, (None, self.num_discrete_actions), name="action_mask")
        self.actions = tf.argmax(
            self.action_mask, 1, name="actions", output_type=tf.int32)
        self.states_input = self.placeholder_dict_from_fields(
            prefix="states", dtype=tf.float32)

    @namescope('predict_actions')
    def predict_actions(self):
        # Compute action from a state: a* = argmax_a Q(s_t,a)
        self.action_prediction_network = self.prediction_network(
            self.states_input)
        self.q_outputs = self.action_prediction_network.q_output

        # Predict actions from Q network
        self.action_scores = tf.identity(
            self.q_outputs, name="action_scores")
        self.predicted_actions = tf.argmax(
            self.action_scores, axis=1, name="predicted_actions", output_type=tf.int32)
        self.predicted_q_values = tf.identity(
            self.q_outputs, name='predicted_q_values')

    @namescope('estimate_future_rewards')
    def estimate_future_rewards(self):
        # Estimate rewards using the next state: r(s_t,a_t) + argmax_a Q(s_{t+1}, a)
        self.gamma = tf.constant(self.discount_factor, shape=(None))

        self.next_states = self.placeholder_dict_from_fields(
            prefix="next_states", dtype=tf.float32)
        self.next_state_mask = tf.placeholder(
            tf.float32, (None,), name="next_state_masks")

        if self.n_step > 1:
            self.actual_n = tf.placeholder(
                tf.float32, (None), name="actual_n")
            self.n_step_done = tf.placeholder(
                tf.float32, (None), name="n_step_done")

            self.n_step_states = self.placeholder_dict_from_fields(
                prefix="n_step_states", dtype=tf.float32)
            self.n_step_state_mask = tf.placeholder(tf.float32, (None),
                                                    name="n_step_state_mask")  # defines if episode done or not

        # calculate target values
        if self.double_q_learning:
            # Reuse Q network for action selection
            next_outputs_network = self.prediction_network(
                self.next_states)

            q_next_outputs = next_outputs_network.q_output

            self.action_selection = tf.argmax(tf.stop_gradient(
                q_next_outputs), 1, name="action_selection_one_step", output_type=tf.int32)
            action_selection_mask = tf.cast(tf.one_hot(
                self.action_selection, self.num_discrete_actions, 1, 0, name='action_selection_mask'), tf.float32)

            # Use target network for action evaluation
            self.target_q_network = self.target_network(self.next_states)
            if self.n_step > 1:
                self.n_step_prediction_network = self.prediction_network(
                    self.n_step_states)
                q_n_step_outputs = self.n_step_prediction_network.q_output
                self.action_selection_n_step = tf.argmax(tf.stop_gradient(
                    q_n_step_outputs), 1, name="action_selection_n_step", output_type=tf.int32)
                action_selection_mask_n_step = tf.cast(tf.one_hot(
                    self.action_selection_n_step, self.num_discrete_actions, 1, 0, name='action_selection_mask_n_step'), dtype=tf.float32)
                self.target_n_step_q_network = self.target_network(
                    self.n_step_states)
                n_step_target_outputs = self.target_n_step_q_network.q_output * \
                    action_selection_mask_n_step
                n_step_target_values = tf.reduce_sum(n_step_target_outputs, axis=[
                    1, ]) * self.n_step_state_mask

            target_outputs = self.target_q_network.q_output * action_selection_mask
            action_evaluation = tf.reduce_sum(
                target_outputs, axis=[1, ], name='action_evaluation')
            target_values = action_evaluation * self.next_state_mask

        else:
            # Initialize target network
            self.target_q_network = self.target_network(self.next_states)
            target_outputs = self.target_q_network.q_output

            # Compute future rewards
            next_action_scores = tf.stop_gradient(target_outputs)
            self.action_selection = tf.argmax(
                next_action_scores, 1, name="action_selection_one_step", output_type=tf.int32)
            target_values = tf.reduce_max(next_action_scores, axis=[
                1, ]) * self.next_state_mask
            tf.summary.histogram("next_action_scores", next_action_scores)

            if self.n_step > 1:
                self.target_n_step_q_network = self.target_network(
                    self.n_step_states)
                n_step_target_outputs = self.target_n_step_q_network.q_output
                next_n_step_action_scores = tf.stop_gradient(
                    n_step_target_outputs)
                self.action_selection_n_step = tf.argmax(
                    next_n_step_action_scores, 1, name="action_selection_n_step", output_type=tf.int32)
                n_step_target_values = tf.reduce_max(next_n_step_action_scores,
                                                     axis=[1, ]) * self.n_step_state_mask

        # calculate final future rewards from next actions weighted with a time factor
        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.future_rewards = self.rewards + self.discount_factor * target_values

        if self.n_step > 1:
            self.n_step_rewards = tf.placeholder(
                tf.float32, shape=[None], name="n_step_rewards")
            self.n_step_future_rewards = self.n_step_rewards + \
                tf.pow(self.gamma, self.actual_n) * n_step_target_values

    @namescope('compute_temporal_differences')
    def compute_temporal_differences(self):
        # loss weights
        weight_loss_one_step = self.params["network"]["loss_one_step_weight"]
        weight_loss_n_step = self.params["network"]["loss_n_step_weight"]
        weight_loss_supervised = self.params["network"]["loss_supervised_weight"]
        reg_param = self.params["network"]["reg_param",
                                           "Amount of L2 weight decay regularisation"]

        # calculate loss and gradients
        self.rho = tf.placeholder(
            tf.float32, (None, self.num_discrete_actions), name="rho")
        td_error, td_loss, _, n_step_td_loss = self._calculate_loss()

        # calculate error for replay priorisation
        # but: use only TD-1 error for calculation!
        self.error = td_error

        # Regularization loss
        prediction_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="prediction_network")
        reg_loss = reg_param * \
            tf.reduce_sum([tf.reduce_sum(tf.square(x))
                           for x in prediction_network_variables])

        # Large margin classification loss
        self.isdemo = tf.placeholder("float", [None])
        action_one_hot = tf.one_hot(
            self.predicted_actions, self.num_discrete_actions, 0.0, self.supervised_margin, dtype='float32')

        loss_class = tf.abs(
            tf.reduce_mean(self.isdemo * (
                tf.reduce_max(self.q_outputs + action_one_hot, axis=[1, ]) - tf.reduce_sum(
                    self.q_outputs * self.action_mask, axis=[1, ])
            ))
        )

        # Compute total loss
        self.loss = weight_loss_one_step * td_loss + \
            reg_loss + weight_loss_supervised * loss_class
        if self.n_step > 1:
            self.loss += weight_loss_n_step * n_step_td_loss

        tf.summary.scalar("td_loss", td_loss)
        if self.n_step > 1:
            tf.summary.scalar("n_step_loss", n_step_td_loss)
        tf.summary.scalar("class_loss", loss_class)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", self.loss)

        self.optimize()

    @namescope('optimizer')
    def optimize(self):
        max_gradient = self.params["network"]["max_gradient",
                                              "Gradient clipping parameter"]

        self.gradients = self.context.optimizer.compute_gradients(self.loss, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="prediction_network"))
        # clip gradients by norm
        with tf.variable_scope('gradient_clipper'):
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(
                        grad, max_gradient), var)
        self.train_op = self.context.optimizer.apply_gradients(
            self.gradients, global_step=self.global_step)

    @namescope('update_target_network')
    def update_target_network(self):
        # Update target network with Q network
        self.target_network_update = []

        # Slowly update target network parameters with Q network parameters
        prediction_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="prediction_network")
        target_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
        for v_source, v_target in zip(prediction_network_variables, target_network_variables):
            # This is equivalent to target = (1-alpha) * target + alpha * source
            update_op = v_target.assign_sub(
                self.target_update_rate * (v_target - v_source))
            self.target_network_update.append(update_op)
        self.target_network_update = tf.group(*self.target_network_update)

    @namescope('summary')
    def create_summaries(self):
        self.hastar_guiding_factor_tf = tf.placeholder(
            "float", shape=(), name="hastar_guiding_factor")
        self.exploration_rate_tf = tf.placeholder(
            "float", shape=(), name="exploration_rate")
        self.per_beta_tf = tf.placeholder(
            "float", shape=(), name="per_beta")
        average_reward_per_batch = tf.reduce_mean(self.rewards)
        demo_sampling_rate_tf = tf.reduce_mean(self.isdemo)

        tf.summary.scalar("hastar_guiding_factor",
                          self.hastar_guiding_factor_tf)
        tf.summary.scalar("exploration_rate",
                          self.exploration_rate_tf)
        tf.summary.scalar("per_beta",
                          self.per_beta_tf)
        tf.summary.scalar("average_reward_in_batch",
                          average_reward_per_batch)
        tf.summary.scalar("demo_sampling_rate", demo_sampling_rate_tf)
        tf.summary.histogram("predicted_q_values", self.predicted_q_values)

    def _calculate_loss(self):
        # compute temporal difference loss
        masked_action_scores = tf.reduce_sum(
            self.action_scores * self.action_mask, axis=[1, ], name='masked_action_scores')
        temp_diff = tf.subtract(masked_action_scores,
                                self.future_rewards, name='temp_diff')
        td_loss = tf.reduce_mean(tf.multiply(tf.transpose(
            self.rho), tf.square(temp_diff)), name='td_loss')

        # N-Step Loss
        n_step_td_loss = None
        n_step_temp_diff = None
        if self.n_step > 1:
            n_step_temp_diff = tf.subtract(
                masked_action_scores, self.n_step_future_rewards, name='n_step_temp_diff')
            n_step_td_loss = tf.reduce_mean(tf.multiply(tf.transpose(
                self.rho), tf.square(n_step_temp_diff)), name='n_step_td_loss')

        return temp_diff, td_loss, n_step_temp_diff, n_step_td_loss

    def q_values(self, observation):
        """
        Get q values for an observation
        """
        observation = self.env_state_to_nn_input(observation)
        tf.reset_default_graph()
        with self.context.graph.as_default():
            return self.context.session.run(
                self.action_prediction_network.q,
                self.fill_session_dict(
                    self.states_input, observation)
            )[0]

    def get_next_best_action(self, observation, explore, i_episode):
        guidance = 0
        observation = self.env_state_to_nn_input(observation)

        if explore and self.guidance is not None and i_episode < self.initialize_guidance_steps:
            action = self.guidance.get_next_best_action(self, observation)
            if action is None:
                action = random.randint(0, self.num_discrete_actions - 1)
            else:
                guidance = 1
        elif explore and self.explorer.explore():
            action = random.randint(0, self.num_discrete_actions - 1)
        else:
            tf.reset_default_graph()

            with self.context.graph.as_default():
                action = self.context.session.run(self.predicted_actions, feed_dict=self.fill_session_dict(self.states_input, observation))[0]

        return action, guidance

    def _get_feed_dict(self, w, experience):
        states, rewards, action_mask, next_states, next_state_mask, done_mask, is_demo, n_step_rewards, n_step_states, n_step_state_mask, n_step_done, actual_n = self.format_experience(
            experience)

        rho = np.transpose(np.tile(w, (self.num_discrete_actions, 1)))

        # build feed dictionary
        feed_dict = {**self.fill_session_dict(self.states_input, states),
                     **self.fill_session_dict(self.next_states, next_states),
                     self.next_state_mask: next_state_mask,
                     self.action_mask: action_mask,
                     self.rewards: rewards,
                     self.rho: rho,
                     self.exploration_rate_tf: float(self.exploration_rate['epsilon']),
                     self.per_beta_tf: float(self.per_beta),
                     self.done_mask: done_mask,
                     self.success_rate_buffer_tf: self.success_rate_buffer,
                     self.agent_steps_buffer_tf: self.agent_steps_buffer,
                     self.episode_return_buffer_tf: self.episode_return_buffer,
                     self.hastar_guiding_factor_tf: self.hastar_guiding_factor,
                     self.isdemo: is_demo
                     }
        if self.n_step > 1:
            feed_dict = {
                **feed_dict,
                **self.fill_session_dict(self.n_step_states, n_step_states),
                self.n_step_rewards: n_step_rewards,
                self.n_step_state_mask: n_step_state_mask,
                self.n_step_done: n_step_done,
                self.actual_n: actual_n,
            }
        return feed_dict, is_demo

    def observe(self, observation, action, reward, next_observation, done, info={}, guided=False, *args, **kwargs):
        import objgraph


        observation = self.env_state_to_nn_input(observation)
        next_observation = self.env_state_to_nn_input(next_observation)

        super().observe(observation=observation, action=action, reward=reward,
                        next_observation=next_observation, done=done, info=info, guided=guided)

        if done:
            self.episode_buffer.clear()

    def update_model(self):
        if not self.params['training']:
            return

        # Check if not enough experiences yet
        if self.replay_buffer.n_entries < self.batch_size:
            return

        idx, _, w, experience = self.retrieve_experience()
        feed_dict, is_demo = self._get_feed_dict(w, experience)

        # whether to calculate summaries
        calculate_summaries = self.training_steps % self.summary_every == 0 and self.summary_writer is not None
        additional_run_args = {} if not self._summary_debug_enabled else {
            'options': self.run_options, 'run_metadata': self.run_metadata}

        errors, cost, _, summary_str = self.context.session.run([
            self.error,
            self.loss,
            self.train_op,
            self.summarize if calculate_summaries else self.no_op
        ], feed_dict=feed_dict, **additional_run_args)

        # update target network using Q-network
        self.context.session.run(self.target_network_update)

        # emit summaries
        if calculate_summaries:
            if self.context.summary_service is not None:
                self.context.summary_service.static_fields['step'] = self.training_steps
                self.context.summary_service.add(
                    self._summary_string_to_dict(summary_str))

            if self._summary_debug_enabled:
                self.summary_writer.add_run_metadata(
                    self.run_metadata, 'step%d' % self.training_steps, self.training_steps)
            self.summary_writer.add_summary(summary_str, self.training_steps)

        self.training_steps += 1
        self._update_replay_buffer(idx, errors, is_demo)

    def _summary_string_to_dict(self, summary_str):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summary_str)
        summaries = {}

        for val in summary_proto.value:
            try:
                list_for_tag = summaries[val.tag]
            except KeyError:
                list_for_tag = []
                summaries[val.tag] = list_for_tag

            # Assuming all summaries are scalars.
            list_for_tag.append(val.simple_value)

        # convert single property arrays to object arrays {a: [1, 2]} -> [{ a: 1 }, { a: 2 }]
        summary_list = None
        for key, item in summaries.items():
            # initialize in first run
            if summary_list is None:
                summary_list = [{} for _ in range(len(item))]

            for idx, val in enumerate(item):
                summary_list[idx][key] = val

        return summary_list

    def format_experience(self, experience):
        states_dict = {}
        next_states_dict = {}
        n_step_states_dict = {}
        for key, val in self.get_nn_inputs_definition().items():
            states_dict[key] = np.zeros(
                (self.batch_size,) + self.get_placeholder_dimension(key)[1:])
            next_states_dict[key] = np.zeros(
                (self.batch_size,) + self.get_placeholder_dimension(key)[1:])
            if self.n_step > 1:
                n_step_states_dict[key] = np.zeros(
                    (self.batch_size,) + self.get_placeholder_dimension(key)[1:])
        rewards = np.zeros((self.batch_size,))
        done_mask = np.zeros((self.batch_size,))
        action_mask = np.zeros(
            (self.batch_size, self.num_discrete_actions) )
        next_state_mask = np.zeros((self.batch_size,))
        is_demo = np.zeros((self.batch_size,))

        n_step_rewards = np.zeros((self.batch_size,))
        n_step_state_mask = np.zeros((self.batch_size,))
        n_step_done = np.zeros((self.batch_size,))
        actual_n = np.zeros((self.batch_size,))

        for k, (s0, a, r, s1, done, demo, n_r, n_s, n_done, n) in enumerate(experience):
            for key, val in self.get_nn_inputs_definition().items():
                states_dict[key][k] = s0[key]
            rewards[k] = r
            action_mask[k][a] = 1
            done_mask[k] = done
            is_demo[k] = demo
            if self.n_step > 1:
                n_step_rewards[k] = n_r
                n_step_done[k] = n_done
                actual_n[k] = n
            # check terminal state
            if not done:
                for key, val in self.get_nn_inputs_definition().items():
                    next_states_dict[key][k] = s1[key]
                    if self.n_step > 1:
                        n_step_states_dict[key][k] = n_s[key]
                next_state_mask[k] = 1
                if self.n_step > 1:
                    n_step_state_mask[k] = 1

        return states_dict, rewards, action_mask, next_states_dict, next_state_mask, done_mask, is_demo, n_step_rewards, n_step_states_dict, n_step_state_mask, n_step_done, actual_n
