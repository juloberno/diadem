# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import os
import json
import numpy as np

import tensorflow as tf
from collections import deque
from abc import ABC, abstractmethod

from diadem.common import BaseObject
from gym import spaces

class Agent(BaseObject):
    def __init__(self, params=None, context=None, *args, **kwargs):
        self.params = params
        self.context = context
        self._initialize_params(params=params)
        self.name = 'agent'

    @abstractmethod
    def initialize(self, *args, **kwargs):
        super().initialize()

        # invalidate saver
        self._saver = None

        self.guidance = kwargs.get('guidance')
        self.summary_service = kwargs.get('summary_service')

        self.log_directory = kwargs.get('log_directory', '.')
        self.checkpoints_directory = kwargs.get('checkpoints_directory', '.')
        self.evaluations_directory = kwargs.get('evaluations_directory', '.')

        # ------- init some internal variable ----------
        self.training_steps = 0  # training step counter
        # saves for the current episode an intermediate buffer, which also allows to use an n-step return, save
        self.episode_buffer = []

        tf.reset_default_graph()
        with self.context.graph.as_default():
            # Create variables
            self.create_variables()

            # merge all summaries
            self.summarize = tf.summary.merge_all()
            self.no_op = tf.no_op()
            var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            # initialize variables
            self.context.session.run(tf.variables_initializer(var_lists))
            # Make sure all variables are initialized
            self.context.session.run(tf.assert_variables_initialized())

            # initialize common summaries for all agents
            self._initialize_summary_writer()

    def _initialize_summary_writer(self):
        # Summary writer init for common variables
        self.summary_writer = tf.summary.FileWriter(
            self.log_directory, self.context.session.graph)

        # enable RAM debugging / summary
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

        self.agent_steps_buffer = deque(
            maxlen=self.running_average_length_steps)
        self.agent_steps_buffer_tf = tf.placeholder(
            tf.float32, shape=[None], name="agent_steps_buffer")

        self.success_rate_buffer = deque(
            maxlen=self.running_average_length_steps)
        self.success_rate_buffer_tf = tf.placeholder(
            tf.float32, shape=[None], name="success_rate_buffer")

        self.episode_return_buffer = deque(
            maxlen=self.running_average_length_steps)
        self.episode_return_buffer_tf = tf.placeholder(
            tf.float32, shape=[None], name="episode_return_buffer")

        self.success_rate_tf = tf.reduce_mean(
            self.success_rate_buffer_tf, name='success_rate')
        self.agent_steps_tf = tf.reduce_mean(
            self.agent_steps_buffer_tf, name='agent_steps')
        self.episode_return_tf = tf.reduce_mean(
            self.episode_return_buffer_tf, name='episode_return')

        tf.summary.scalar("success_rate", self.success_rate_tf)
        tf.summary.scalar("agent_steps", self.agent_steps_tf)
        tf.summary.scalar("episode_return", self.episode_return_tf)

        self.summarize = tf.summary.merge_all()

        if self.summary_writer is not None:
            # Graph was not available when journalist was created
            self.summary_writer.add_graph(self.context.graph)

    def param_annealing(self, current_step):
        self.params.anneal(current_step=current_step)

    @abstractmethod
    def _initialize_params(self, params=None):
        super()._initialize_params(params)

        self.running_average_length_steps = self.params["summary"]["running_average",
                                                                   "Describes the window length to build running averages for steps, success rate and episode return"]
        # TODO: change summary_every
        self.summary_every = self.params["summary"]["summary_every"]

        self._observe_via_agent_manager = self.params["observe_via_agent_manager"]

    @abstractmethod
    def create_variables(self):
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    @abstractmethod
    def observe(self, observation, action, reward, next_observation, done, info={}, guided=False):
        import objgraph

        self.episode_buffer.append(
            [observation, action, reward, next_observation, done, guided])

        self._add_to_summary('done', done)
        self._add_to_summary('reward', reward)
        self._add_to_summary('action', action)
        self._add_to_summary('goal_reached', info.get('goal_reached', False))

        if done:
            ret = 0
            for x in self.episode_buffer:
                ret += x[2]
            self.episode_return_buffer.append(ret)
            if "goal_reached" in info:
                self.success_rate_buffer.append(info["goal_reached"])
            self.agent_steps_buffer.append(len(self.episode_buffer))

    def _add_to_summary(self, name, value):
        if self.context.summary_service is not None:
            self.context.summary_service.static_fields[name] = value

    @abstractmethod
    def get_next_best_action(self, observation, explore, **kwargs):
        return  # action, is_guided

    @abstractmethod
    def update_model(self):
        self.training_steps += 1

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self.params['max_kept_checkpoints'] or 0)
        return self._saver

    @property
    def observe_via_agent_manager(self):
        return self._observe_via_agent_manager

    def freeze(self, global_step):
        if not self.params['training']:
            return

        checkpoints_directory = self.checkpoints_directory
        if not os.path.exists(checkpoints_directory):
            os.makedirs(checkpoints_directory, exist_ok=True)

        tf.reset_default_graph()
        with self.context.graph.as_default():
            save_path = self.saver.save(
                self.context.session,
                os.path.join(checkpoints_directory, 'agent'),
                global_step=global_step
            )
        self.log.info("Freeze to " + save_path)

        with open(os.path.join(checkpoints_directory, 'agent_checkpoint.json'), "w+") as json_file:
            json_file.write(json.dumps({
                'global_step': global_step,
                'path': save_path
            }))

    def reset(self):
        self.context.reset()

    def close(self):
        self.context.close()

    def restore(self):
        file_path = os.path.join(
            self.checkpoints_directory, 'agent_checkpoint.json')
        # self.log.info("try to store from path: " + file_path)
        self.log.info("try to store from path: " + file_path)

        tf.reset_default_graph()
        with self.context.graph.as_default():
            if os.path.isfile(file_path):
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.loads(json_file.read())
                    except:
                        self.training_steps = 1
                        self.context.session.run(
                            tf.global_variables_initializer())
                        return

                    checkpoint_path = os.path.join(
                        self.checkpoints_directory, data.get('path').split('/')[-1])
                    self.saver.restore(
                        self.context.session, checkpoint_path)
                    self.log.info("Restore successful")
                    self.training_steps = self.global_step.eval(
                        self.context.session)
                    self.context.session.run(tf.trainable_variables())

            else:
                self.training_steps = 1
                self.context.session.run(tf.global_variables_initializer())

    def get_nn_inputs_definition(self):
        """ Wrap space definition of gym spaces, to have the possibility to later extend to more complex spaces
        
        Returns:
            [dict] -- [each key becomes one tf.placeholder later on whith the given dimensions ]
        """
        if isinstance(self.context.environment.observationspace, spaces.Box):
            return {"env_state" : np.array(self.context.environment.observationspace.shape)} # 1D state
        else:
            raise ValueError("Unsupported observation space definition.")

    def env_state_to_nn_input(self, env_state):
        data = None
        if isinstance(env_state, np.ndarray):
            data = np.array(env_state)
        data = data.flatten()

        if self.context.preprocessor:
            data = self.context.preprocessor.preprocess(data)

        return {"env_state": data}

    def get_placeholder_dimension(self, field_name):
        field_dim = self.get_nn_inputs_definition()[field_name]
        if isinstance(field_dim, np.ndarray):
            tp = (None,)  # for nn batch
            for i in np.nditer(field_dim):
                tp += (i,)
        else:
            tp = (None, field_dim)
        return tp

    def placeholder_dict_from_fields(self, prefix, dtype):
        placeholder = {}
        fields = self.get_nn_inputs_definition()
        for key in fields.keys():
            placeholder[key] = tf.placeholder(
                dtype=dtype, shape=self.get_placeholder_dimension(key), name=prefix + "_" + key)
            
        return placeholder

    def fill_session_dict(self, placeholders, data):
        dict = {}

        for key, value in placeholders.items():
            temp = data[key]
            if not isinstance(data[key], np.ndarray):
                temp = np.array(temp)
            if temp.shape[0] == self.get_placeholder_dimension(key)[1]:
                # we are not processing a batch data as first data dimension is equal to converted dimension
                # -> add the batch dimension
                temp = np.array([temp])

            dict[value] = temp
        return dict
