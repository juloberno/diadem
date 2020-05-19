# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import datetime
import glob
from collections import deque
import time
import numpy as np
import json
import logging

from diadem.common import config_logging
from diadem.common import BaseObject

class Experiment(BaseObject):
    def __init__(self, main_dir, params=None, context=None, agent=None, visualizer=None):
        super().__init__()
        self.context = context
        self.agent = agent
        self.main_dir = main_dir
        self.visualizer = visualizer

        self._initialize_params(params=params)
        self.initialize()

    def _initialize_params(self, params=None):
        super()._initialize_params(params=params)

        self.max_episodes = self.params["max_episodes"]
        self.num_cycles = self.params["num_cycles"]
        self.freeze_every_num_environment_steps = self.params["freeze_every_num_environment_steps"]
        self.freeze_every_num_episodes = self.params["freeze_every_num_episodes"]
        self.restore_on_start = self.params["restore_on_start"]
        
        self.update_model_every_num_episodes = self.params["update_model_every_num_episodes"]
        self.update_model_every_num_steps = self.params[
            "update_model_every_num_steps", "Integer, determines update frequency of model."]
        self.anneal_over = self.params["anneal_over",
                                       "String, Determines if annealing is done over [steps] or [episodes]"]
        self.final_evaluation_on_test = self.params["final_evaluation_on_test"]
        self.final_evaluation_on_train = self.params["final_evaluation_on_train"]
        self.final_evaluation_batch_size = self.params["final_evaluation_batch_size"]
        self.tmp_evaluation_every = self.params["tmp_evaluation_every"]
        self.tmp_evaluation_batch_size = params["tmp_evaluation_batch_size"]
        self.tmp_evaluation_on_test = self.params["tmp_evaluation_on_test"]
        self.tmp_evaluation_on_train = self.params["tmp_evaluation_on_train"]
        self.experiment_name = self.params["name"]
        self.run_on_gluster = self.params["run_on_gluster"]
        self.log_level = eval("logging.{}".format(self.params["log_level", "Log levels allowed by python logging module, e.g. ""DEBUG"", ""INFO"",... "]))
        self.max_agent_steps = self.params["max_agent_steps"]
        self.action_timedelta = self.params["action_timedelta",
                                            "Specifies the amount of time the agent's actions is applied to the environment."]
        self.additional_agent_kwargs = self.params["additional_agent_kwargs"] or {}
        self._has_memory_analysis = self.params["analyse_memory", "Specifies if memory consumption should be printed or not"]
        self._select_random_train_sample = self.params["random_train_sample", "Specifies if train sample should be selected randomly or incrementally otherwise"]


    def run(self):
        # Main loop for runs
        for i_cycle in range(1, self.num_cycles + 1):
            # Initialization
            if not os.path.exists(self._get_checkpoint_folder(i_cycle)):
                os.makedirs(self._get_checkpoint_folder(i_cycle))

            if not os.path.exists(self._get_log_file_folder(i_cycle)):
                os.makedirs(self._get_log_file_folder(i_cycle))

            if self.run_on_gluster:
                config_logging(log_level=self.log_level, console=False, filename=self._get_pylogging_file_name(i_cycle))
            else:
                config_logging(log_level=self.log_level)

            self.agent.initialize(
                evaluations_directory=self._get_evaluations_file_folder(
                    i_cycle),
                log_directory=self._get_log_file_folder(i_cycle),
                checkpoints_directory=self._get_checkpoint_folder(i_cycle),
                **self.additional_agent_kwargs
            )
            
            self.log.info("Starting with training cycle {}".format(i_cycle))

            # saver is created after td-variable creation
            if self.restore_on_start:
                self.restore(i_cycle)
            else:
                self.start_episode = 0

            summary_filename = self.params['summary_file'] or 'train.csv'
            summary_path = os.path.join(
                self._get_log_file_folder(i_cycle), summary_filename)

            update_count = 0

            tf.get_default_graph().finalize()

            start_time = time.time()
            # Train until maximum number of training episodes is reached
            i_episode = self.start_episode
            for i_episode in range(self.start_episode, self.max_episodes + 1):
                self.log.info("Running episode {} of {}".format(i_episode, self.max_episodes))
                self.context.summary_service.static_fields['episode'] = i_episode

                # Perform one episode
                observation, _ = self._get_sample()

                done = False
                step_count = 0


                self.context.summary_service.static_fields['step'] = 0

                # perform interactions in the environment with agent, we convert between environment and neural network state and action definitions
                while step_count < self.max_agent_steps and not done:
                    self.context.summary_service.static_fields['state'] = self.context.environment.state_as_string()
                    self.context.summary_service.static_fields['step'] = step_count

                    # get action
                    env_action, is_guided = self.agent.get_next_best_action(
                        observation=observation, explore=self.params['explore'], i_episode=i_episode)

                    # step environment
                    next_observation, reward, done, info = self.context.environment.step(
                        env_action)

                    # let agent observe reward
                    self.agent.observe(observation=observation, action=env_action, reward=reward, next_observation=next_observation,
                                       done=done, info=info, guided=is_guided)
                    # change to next state
                    observation = next_observation
                    step_count += 1
                    update_count += 1

                    if update_count % self.update_model_every_num_steps == 0:
                       self.agent.update_model()

                    self.log.debug('---------------------')
                    self.log.debug('Action: ' + str(env_action))
                    self.log.debug('Step count: ' +
                            str(step_count))
                    self.log.debug('Step Reward: ' + str(reward))

                    if self.visualizer is not None:
                        self.visualizer.visualize(episode=i_episode, step=step_count, last_step=(done or step_count == self.max_agent_steps))

                    summary_every = self.context.agent_manager.main_agent.params['summary']['summary_every']
                    if summary_every > 0 and update_count % summary_every == 0:
                        self.context.summary_service.add({})

                    if self.freeze_every_num_environment_steps > 0 and update_count % self.freeze_every_num_environment_steps == 0:
                        self.store_model(i_cycle, i_episode,
                                         update_count, update_with_episode=False)
                        log_string = self.context.summary_service.dump(summary_path)
                        self.log.info("Dumped summary to {}".format(os.path.abspath(summary_path)))
                        self.log.info("----------------- Summary --------------------- \n {}".format(log_string))

                    if done:
                        break

                if i_episode % self.update_model_every_num_episodes == 0 and update_count % self.update_model_every_num_steps != 0:
                    self.agent.update_model()

                if self.anneal_over == "steps":
                    self.agent.param_annealing(current_step=update_count)
                elif self.anneal_over == "episodes":
                    self.agent.param_annealing(current_step=i_episode)

                # Save model and information when new checkpoint is reached
                if self.freeze_every_num_episodes > 0 and i_episode % self.freeze_every_num_episodes == 0:
                    self.store_model(i_cycle, i_episode, update_count)
                    log_string = self.context.summary_service.dump(summary_path)
                    self.log.info("Dumped summary to {}".format(os.path.abspath(summary_path)))
                    self.log.info("----------------- Summary --------------------- \n {}".format(log_string))

                # Print information about episode and progress if not on gluster
                self.log.debug("-----------")
                self.log.debug(self.experiment_name + " | Cycle: " +
                        str(i_cycle) + " | Episode: " + str(i_episode))
                # if isinstance(self.agent, HbaAgent):
                self.log.debug('Took ' + str(time.time()-start_time) +
                        ' seconds to finish')
                self.log.debug(
                    "Finished after {} timesteps".format(step_count + 1))
                start_time = time.time()

                # Do intermediate evaluation of temporary model
                if self.tmp_evaluation_every > 0 and (i_episode == 1 or i_episode % self.tmp_evaluation_every == 0):
                    self.evaluation(i_cycle, i_episode,
                                    update_count, is_final=False)

            self.context.summary_service.add({})
            self.context.summary_service.dump(summary_path)
            self.context.summary_service.reset()

            # Save final train model for this run
            self.store_model(i_cycle, i_episode, update_count)

            # Evaluate final model
            self.evaluation(i_cycle, i_episode, update_count, is_final=True)

            # Reset agent for next run
            self.agent.reset()

        # close all sessions finally and open handlers
        self.agent.close()

    def store_model(self, cycle, episode, update_count, update_with_episode=True):
        if not self.params['training']:
            return

        global_step = episode if update_with_episode else update_count
        checkpoints_folder = self._get_checkpoint_folder(cycle)

        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder, exist_ok=True)

        checkpoint_file = self.params['checkpoint_file'] or 'checkpoint.json'
        with open(os.path.join(checkpoints_folder, checkpoint_file), "w+") as json_file:
            json_file.write(json.dumps({
                'episode': episode,
                'step': update_count
            }))

        self.agent.freeze(global_step)
        self.log.info("Freezed episode %i and step %i" %
                      (episode, update_count))

    def restore(self, cycle):
        checkpoint_file = self.params['checkpoint_file'] or 'checkpoint.json'
        file_path = os.path.join(
            self._get_checkpoint_folder(cycle), checkpoint_file)
        self.log.info("try to store from path: " + file_path)

        if os.path.isfile(file_path):
            with open(file_path, 'r') as json_file:
                try:
                    data = json.loads(json_file.read())
                    self.start_episode = data.get('episode')
                except:
                    self.start_episode = 1
        else:
            self.start_episode = 1

        self.agent.restore()

    def _get_evaluations_file_folder(self, cycle):
        return os.path.join(self.main_dir, "training", "evaluations", "cycle_" + str(cycle))

    def _get_checkpoint_path(self, cycle):
        return os.path.join(self._get_checkpoint_folder(cycle), self._get_checkpoint_name(cycle), "./")

    def _get_log_file_folder(self, cycle):
        return os.path.join(self.main_dir, "training", "logs", "cycle_" + str(cycle))

    def _get_checkpoint_folder(self, cycle):
        return os.path.join(self.main_dir, "training", "checkpoints", "cycle_" + str(cycle))

    def _get_checkpoint_name(self, cycle):
        return './' + self.experiment_name

    def _get_pylogging_file_name(self, cycle):
        return os.path.join(self._get_log_file_folder(cycle), "log")

    def _get_latest_cycle(self):
        """
        Returns the checkpoint folder of the latest cycle. It returns None if no checkpoints was previously created
        """
        latest_cycle = 0
        folder = None
        for i in range(1, self.num_cycles + 1):
            checkpoint_folder = self._get_checkpoint_folder(i)
            if os.path.exists(checkpoint_folder):
                folder = checkpoint_folder
                latest_cycle = i
            else:
                break
        return latest_cycle, folder

    def _get_sample(self):
        init_error = True
        if not self.context.environment.contains_training_data:
            if self.context.datamanager is None:
                self.log.error("No datamanager given")
                return
            while init_error:
                # get the next full state of the environment from which episode starts
                sample = self.context.datamanager.get_random_train_sample() if self._select_random_train_sample else self.context.datamanager.get_next_train_sample()
                self.context.summary_service.static_fields['sample_idx'] = self.context.datamanager.current_train_idx
                observation, init_error = self.context.environment.reset(
                    sample)
        else:
            observation, init_error = self.context.environment.reset()
        return observation, init_error

    def evaluation(self, i_cycle, i_episode, i_step, is_final=False):
        df_folder = self._get_evaluations_file_folder(i_cycle)
        if not os.path.exists(df_folder):
            os.makedirs(df_folder, exist_ok=True)
        df_path = os.path.join(df_folder, "evaluation.csv")
        if self.context.datamanager is None and self.context.environment.contains_training_data():
            self.environment_data_evaluation(i_episode, i_step, is_final)
        elif self.context.datamanager:
            self.train_test_evaluation(i_episode, i_step, is_final)
        summary = self.context.summary_service
        summary.dump(df_path)

    def environment_data_evaluation(self, i_episode, i_step, is_final=False):
        batch_size = self.final_evaluation_batch_size if is_final else self.tmp_evaluation_batch_size
        success_rate_buffer = []
        collision_rate_buffer = []
        max_steps_rate_buffer = []
        average_return_buffer = []
        average_steps_buffer = []

        for i in range(0, batch_size):
            if not self.run_on_gluster:
                i += 1
                msg = 'Train: ' + str(i) + ' / ' + \
                    str(batch_size) + ' evaluated!'
                print(msg, end='\r')
            observation, _ = self.context.environment.reset()
            total_rewards = 0.0

            # Perform one episode
            for t in range(self.max_agent_steps):
                action, _ = self.agent.get_next_best_action(
                    observation, False, i_episode)
                next_observation, reward, done, info = self.context.environment.step(
                    action)
                goal_reached = info['goal_reached']
                total_rewards += reward
                observation = next_observation

                if done:
                    break

            if not done:
                max_steps_rate_buffer.append(1)
            else:
                max_steps_rate_buffer.append(0)

            average_steps_buffer.append(t)
            success_rate_buffer.append(int(goal_reached))
            collision_rate_buffer.append(int(done and not goal_reached))
            average_return_buffer.append(total_rewards)

        success_rate = np.mean(success_rate_buffer)
        collision_rate = np.mean(collision_rate_buffer)
        max_steps_rate = np.mean(max_steps_rate_buffer)
        average_return = np.mean(average_return_buffer)
        average_steps = np.mean(average_steps_buffer)
        exploration_rate = self.agent.explorer.epsilon.val if self.agent.explorer else 0
        priorisation_beta = self.agent.per_beta.val if self.agent.per_beta else 0

        summary = self.context.summary_service
        summary.add({
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "max_steps_rate": max_steps_rate,
            "average_return": average_return,
            "average_steps": average_steps,
            "episode": i_episode,
            "step": i_step,
            "sample_idx": "resets",
            "exploration_rate": exploration_rate,
            "priorisation_beta": priorisation_beta,
            **self._sysinfo
        })


    def train_test_evaluation(self, i_episode, i_step, is_final=False):
        if self.context.datamanager is None:
            self.log.error("Could not find datamanager for train/test evaluation")
            return

        if is_final:
            on_train = self.final_evaluation_on_train
            on_test = self.final_evaluation_on_test
        else:
            on_train = self.tmp_evaluation_on_train
            on_test = self.tmp_evaluation_on_test

        if not self.run_on_gluster:
            print("----------------------")
            print("Evaluation.." if not is_final else "Final evaluation..")
            print("-")

        batch_size = self.final_evaluation_batch_size if is_final else self.tmp_evaluation_batch_size
        batches = {}
        batches_idx = {}
        if on_train:
            batches["train"] = self.context.datamanager.get_random_train_batch(
                batch_size)
            batches_idx["train"] = self.context.datamanager.current_train_batch_idx
        if on_test:
            batches["test"] = self.context.datamanager.get_random_test_batch(
                batch_size)
            batches_idx["test"] = self.context.datamanager.current_test_batch_idx

        for batch_name, batch in batches.items():
            success_rate_buffer = []
            collision_rate_buffer = []
            max_steps_rate_buffer = []
            average_return_buffer = []
            average_steps_buffer = []

            i = 0
            for s in batch:
                if not self.run_on_gluster:
                    i += 1
                    msg = 'Train: ' + str(i) + ' / ' + \
                        str(batch_size) + ' evaluated!'
                    print(msg, end='\r')
                observation, _ = self.context.environment.reset(s)
                total_rewards = 0.0

                # Perform one episode
                for t in range(self.max_agent_steps):
                    action, _ = self.agent.get_next_best_action(
                        observation, False, i_episode)
                    next_observation, reward, done, info = self.context.environment.step(
                        action)
                    goal_reached = info['goal_reached']
                    total_rewards += reward
                    observation = next_observation

                    if done:
                        break

                if not done:
                    max_steps_rate_buffer.append(1)
                else:
                    max_steps_rate_buffer.append(0)

                average_steps_buffer.append(t)
                success_rate_buffer.append(int(goal_reached))
                collision_rate_buffer.append(int(done and not goal_reached))
                average_return_buffer.append(total_rewards)

            success_rate = np.mean(success_rate_buffer)
            collision_rate = np.mean(collision_rate_buffer)
            max_steps_rate = np.mean(max_steps_rate_buffer)
            average_return = np.mean(average_return_buffer)
            average_steps = np.mean(average_steps_buffer)
            exploration_rate = self.agent.explorer.epsilon.val if self.agent.explorer else 0
            priorisation_beta = self.agent.per_beta.val if self.agent.per_beta else 0

            summary = self.context.summary_service
            summary.add({
                "batch_name" : batch_name,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "max_steps_rate": max_steps_rate,
                "average_return": average_return,
                "average_steps": average_steps,
                "episode": i_episode,
                "step": i_step,
                "sample_idx": json.dumps(batches_idx[batch_name]),
                "exploration_rate": exploration_rate,
                "priorisation_beta": priorisation_beta,
                **self._sysinfo
            })

    @property
    def _sysinfo(self):
        import psutil
        import time
        pid = os.getpid()
        py = psutil.Process(pid)
        memory_use = py.memory_info()[0]/2.**30

        return {
            'memory': memory_use,
            'time': time.time(),
            'time_hr': time.strftime("%Y-%m-%d %H:%M"),
            'cpu': psutil.cpu_percent()
        }
