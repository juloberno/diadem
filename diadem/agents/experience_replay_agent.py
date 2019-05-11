# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import numpy as np

from diadem.agents.agent import Agent
from diadem.agents.buffer import Prioritized


class ExperienceReplayAgent(Agent):
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        # Create replay buffer
        self.replay_buffer = kwargs.get("buffer", Prioritized(
            self.replay_buffer_size, self.demo_capacity_factor))
        self.store_experience_cnt = 0

    def _initialize_params(self, params=None):
        super()._initialize_params(params)

        self.prio_max = 0
        self.discount_factor = self.params["discount_factor",
                                           "This param gives the discount factor of the underlying MDP"]
        self.n_step = self.params["n_step"]
        self.batch_size = self.params["network"]["batch_size"]
        self.replay_buffer_size = self.params["prioritizing"]["replay_buffer_size"]
        self.demo_capacity_factor = self.params["prioritizing"]["demo_capacity_factor"]

        self.per_epsilon_alpha = self.params["prioritizing"]["per_epsilon_alpha"]
        self.per_epsilon_demo = self.params["prioritizing"]["per_epsilon_demo"]
        # small positive constant that ensures that no transition has zero priority
        self.per_alpha = self.params["prioritizing"]["per_alpha"]
        self.per_beta = self.params["prioritizing"]["per_beta",
                                                    "Specifies the beta value for prioritization", "Lin"]

        self.store_replay_every = self.params["prioritizing"]["store_replay_every", ""]

    def retrieve_experience(self):
        idx = None
        priorities = None
        w = None

        idx, priorities, experience = self.replay_buffer.sample(
            self.batch_size)

        sampling_probabilities = priorities / self.replay_buffer.total()
        beta = float(self.per_beta)
        w = np.power(self.replay_buffer.n_entries *
                     sampling_probabilities, -beta)
        w = w / w.max()
        return idx, priorities, w, experience

    def _store_experience(self, transition):
        _, _, _, _, _, is_demo, _, _, done, _ = transition

        # Calculate priority and add experience to memory (always store end states)
        if self.store_experience_cnt % self.store_replay_every == 0 or done:

            if is_demo:
                epsilon = self.per_epsilon_demo
            else:
                epsilon = self.per_epsilon_alpha

            priority = max(self.prio_max, epsilon)
            experience = (transition)
            if is_demo:
                self.replay_buffer.add_demo(priority, experience)
            else:
                self.replay_buffer.add(priority, experience)

        self.store_experience_cnt += 1

    def error2priority(self, errors, is_demo):
        errors = np.abs(errors)
        sums = []
        for i in range(0, len(errors)):
            if is_demo[i]:
                sums.append(errors[i] + self.per_epsilon_demo)
            else:
                sums.append(errors[i] + self.per_epsilon_alpha)

        return np.power(sums, self.per_alpha)

    def _update_replay_buffer(self, idx, errors, is_demo):
        # Update Priorities in experience sum tree for batch
        priorities = self.error2priority(errors, is_demo)
        for i in range(0, self.batch_size):
            self.replay_buffer.update(idx[i], priorities[i])
        self.prio_max = max(priorities.max(), self.prio_max)

    def observe(self, observation, action, reward, next_observation, done, info={}, guided=False):
        super().observe(observation=observation, action=action, reward=reward,
                        next_observation=next_observation, done=done, info=info, guided=guided)

        added_experiences = 0

        # extend the episode buffer with n-step return, n-step-state (use as target in Q(n_step_state, . )), n_step_done and n
        if self.n_step > 1 and len(self.episode_buffer) >= self.n_step:
            # could be more efficient, but this way, easier to understand
            n_step_reward = sum([e[2]*self.discount_factor**i for i,
                                 e in enumerate(self.episode_buffer[-self.n_step:])])
            n_step_state = next_observation
            n_step_done = done
            n = self.n_step

            self.episode_buffer[-n].extend(
                [n_step_reward, n_step_state, n_step_done, n])
            self._store_experience(self.episode_buffer[-n])
            added_experiences += 1
        elif self.n_step <= 1:
            self.episode_buffer[-1].extend([0, None, done, 1])

        # if episode done set-n-step return of the last samples
        if done:
            if self.n_step > 1 and len(self.episode_buffer) >= self.n_step:
                # remove first transition as n-step already set
                remaining_transitions = self.episode_buffer[-self.n_step + 1:]
            else:
                remaining_transitions = self.episode_buffer

            if self.n_step > 1:
                remaining_transitions = self._set_n_step_when_done(
                    remaining_transitions, self.n_step)

            for e in remaining_transitions:
                self._store_experience(e)

            added_experiences += len(remaining_transitions)

        return added_experiences

    def _set_n_step_when_done(self, container, n):
        t_list = list(container)
        # accumulated reward of first (trajectory_n-1) transitions
        n_step_reward = sum(
            [t[2] * self.discount_factor ** i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
        for begin in range(len(t_list)):
            end = min(len(t_list) - 1, begin + self.n_step - 1)
            n_step_reward += t_list[end][2] * \
                self.discount_factor ** (end - begin)
            # extend[n_reward, n_next_s, n_done, actual_n]
            t_list[begin].extend(
                [n_step_reward, t_list[end][3], t_list[end][4], end - begin + 1])
            n_step_reward = (
                n_step_reward - t_list[begin][2]) / self.discount_factor
        return t_list
