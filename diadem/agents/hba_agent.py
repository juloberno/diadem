# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.agents.agent import Agent
from collections import deque
import numpy as np
import json
from scipy.stats import norm


def _linear(x, weight=1, bias=0):
    return weight * x + bias


def _gauss_policy(name, policy_params):
    def _gauss(ground_truth_action, action):
        # mean = _linear(action_hypo, **policy_params['mean'])
        # deviation = _linear(action_hypo, **policy_params['deviation'])
        # return norm(mean, deviation).pdf(action_hba)
        mean = _linear(ground_truth_action, **policy_params['mean'])
        deviation = _linear(ground_truth_action, **policy_params['deviation'])
        prob = norm.pdf(action, mean, deviation)
        return prob

    return _gauss


class HbaAgent(Agent):
    def __init__(self, context = None, params=None, *args, **kwargs):
        super().__init__(context=context, params=params)

        self.name = 'hba_agent'
        self.prior = {}
        self.policy_types_history = {}
        self.hypothesises = {}

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.discount_factor = self.params['discount_factor']
        self.is_first_action = True
        self.hypo_aliases = {
            self.name: self.name
        }
        for name, policy_params in self.params['hypothesises'].items():
            policy_params = policy_params.convert_to_dict()
            if policy_params['type'] == 'gauss':
                policy = _gauss_policy(name, policy_params)
            else:
                raise ValueError('Invalid agent type policy')

            self.hypo_aliases[name] = policy_params.get('name', name)
            self.hypothesises[name] = {
                'agent': self.context.agent_manager.agent(name),
                'name': self.hypo_aliases[name],
                'policy': policy
            }

        self.num_participants = len(self.context.environment.create_init_states()['states'])

        self.agent_finished = np.zeros([self.num_participants])
        self._add_to_summary('first_step', 1)
        for agent in range(self.num_participants):
            self._add_to_summary('finished_' + str(agent), 0)

        self.clear_history()

    def _initialize_params(self, params=None):
        super()._initialize_params(params)

        self.hypo_q_values = {}

        self.max_history_items = params["max_history_items"]
        if self.max_history_items < 1:
            raise ValueError('History must be at least 1')
        self.history = deque(maxlen=self.max_history_items)

        self.num_hypothesises = len(params["hypothesises"])

    def clear_history(self):
        self.agent_action_prob_history = {
            agent: {
                name: deque(maxlen=self.max_history_items) for name in self.hypothesises
            }
            for agent in range(self.num_participants)
        }
        self.last_posterior = {
            agent: {
                name: 1 for name in self.hypothesises
            }
            for agent in range(self.num_participants)
        }

        for name in self.params["hypothesises"]:
            self.prior[name] = self.params['hypothesises'][name]['prior'] or 1 / self.num_hypothesises


    def observe(self, observation, action, reward, next_observation, done, info={}, guided=False):
        super().observe(observation=observation, action=action, reward=reward,
                        next_observation=next_observation, done=done, guided=guided)

        self._add_to_summary('first_step', self.is_first_action)
        for agent, finished in enumerate(self.agent_finished):
            self._add_to_summary('finished_' + str(agent), finished)
        self._memorize(observation, action)

        if done:
            self.episode_buffer.clear()
            self.is_first_action = True
            self.agent_finished[:] = 0
            if self.params['clear_history_each_episode']:
                self.clear_history()
        else:
            self.is_first_action = False

    def get_next_best_action(self, observation, *args, **kwargs):        
        def _hypo_reward(agent):
            q_values = agent.q_values(observation)
            p_values = agent.p_values(observation)
            for action, action_q_value in enumerate(q_values):
                self.context.summary_service.static_fields['q_value_%s_%i' % (agent.name, action)] = action_q_value
            for action, action_p_values in enumerate(p_values):
                self.context.summary_service.static_fields['p_values_%s_%i' % (agent.name, action)] = json.dumps(action_p_values.tolist())
            return q_values

        hypo_rewards = {name: _hypo_reward(
            hypothesis['agent']) for name, hypothesis in self.hypothesises.items()}

        for name, agent in self.context.agent_manager.agents.items():
            self._add_to_summary('env_hypo_' + self.hypo_aliases[name], agent.context.environment.current_hypothesis)

        # now set the state to the ground truth environment to get the
        # ground truth actions of the other agents
        actions = [state[-1][1] for state in observation.values()]
        if self.is_first_action:
            env_state = [states[-1][0] for states in observation.values()]
            env = self.context.environment
            init_state = self.env_state_to_init_state(
                env_state)
            env.set_state(init_state)
            state, _, _, _ = env.step(0)
            actions = [state[-1][1] for state in state.values()]

        for idx, action in enumerate(actions):
            if idx == 0:
                continue

            other_agent_idx = idx - 1   # ignore ego vehicle
            if action == 0:
                self.agent_finished[other_agent_idx] = 1
                self._add_to_summary('finished_' + str(other_agent_idx), 1)

        rewards = []

        posteriors = {hypothesis: self.posterior(
            hypothesis) for hypothesis in self.hypothesises}
        posteriors_percent = {}
        for hypo, posterior in posteriors.items():
            posteriors_percent[hypo] = posterior / sum(posteriors.values())
            self._add_to_summary('posterior_percent_' +
                                 self.hypo_aliases[hypo], posteriors_percent[hypo])

        for action in range(self.num_discrete_actions):
            if self.params['use_p_values']:
                rewards.append(
                    sum([posteriors_percent[hypothesis] * hypo_rewards[hypothesis][action] for hypothesis in self.hypothesises]))
            else:
                rewards.append(
                    sum([posteriors_percent[hypothesis] * hypo_rewards[hypothesis][action] for hypothesis in self.hypothesises]))
            self._add_to_summary('q_value_hba_' + str(action), rewards[-1])
        action = self.nn_action_to_env_action(
            np.argmax(rewards))
        return action, 0

    def agent_posterior(self, agent, hypothesis):
        posterior_sum = sum(
            [self.agent_hypo_likelihood(agent, hypo) * self.prior[hypo] for hypo in self.hypothesises]
        )
        if posterior_sum > 0:
            posterior = self.agent_hypo_likelihood(agent, hypothesis) * self.prior[hypothesis] / posterior_sum
        else:
            posterior = self.last_posterior[agent][hypothesis]
        
        # use latest posterior if agent already finished
        if self.is_first_action or self.agent_finished[agent]:
            posterior = self.last_posterior[agent][hypothesis]

        self.last_posterior[agent][hypothesis] = posterior

        self._add_to_summary('posterior_' + str(agent) +
                             "_" + self.hypo_aliases[hypothesis], posterior)
        return posterior

    def posterior(self, hypothesis):
        posterior = np.prod([self.agent_posterior(agent, hypothesis)
                             for agent in self.agent_action_prob_history])
        self._add_to_summary('posterior_' + self.hypo_aliases[hypothesis], posterior)
        return posterior

    def agent_hypo_likelihood(self, agent, hypothesis):
        history = self.agent_action_prob_history[agent][hypothesis]
        cur_history_length = len(history)

        def discount_factor(time):
            return self.discount_factor**(cur_history_length-time-1)

        if self.params['posterior_type'] == 'sum':
            likelihood = np.sum(
                [discount_factor(t) * action_prob for t, action_prob in enumerate(history)]
            )
        else:
            likelihood = np.prod(
                [discount_factor(t) * action_prob for t, action_prob in enumerate(history)]
            )
        
        self._add_to_summary('likelihood_' + str(agent) +
                             '_' + self.hypo_aliases[hypothesis], likelihood)
        return likelihood

    def _memorize(self, s_t, a_t):
        """
        Memorize a historical item in the following format:
        ((s1, a1, ...)
        """
        self.history.append((s_t, a_t))

        nn_env = self.convert({agent: [states[-1]] for agent, states in s_t.items()})
        self._add_to_summary('x_ego', nn_env[0])
        self._add_to_summary('y_ego', nn_env[1])
        self._add_to_summary('v_ego', nn_env[2])
        
        def latest_agent_action(state, agent):
            return state[agent][-1][1]

        # restore latest environment agent states and actions from history state
        env_state = [states[-1][0] for states in s_t.values()]

        # now set the state to the ground truth environment to get the
        # ground truth actions of the other agents
        env = self.context.environment
        init_state = self.env_state_to_init_state(env_state)
        env.set_state(init_state)
        ground_truth_state, _, _, _ = env.step(a_t)

        for agent in range(1, len(env_state)):
            ground_truth_action = latest_agent_action(
                ground_truth_state, agent)
            real_action = latest_agent_action(s_t, agent)

            self._add_to_summary(
                'gt_action_' + str(agent-1), ground_truth_action)
            self._add_to_summary('real_action_' + str(agent-1), real_action)

            for name, hypothesis in self.hypothesises.items():
                policy = hypothesis['policy']
                action_prob = policy(ground_truth_action, real_action)
                if not self.is_first_action and not self.agent_finished[agent-1]:
                    self.agent_action_prob_history[agent-1][name].append(
                        action_prob)

    @property
    def explorer(self):
        return None

    @property
    def per_beta(self):
        return None


if __name__ == '__main__':
    import unittest
    from diadem import Params

    class MockTypeAgent:
        def __init__(self, mock_get_action, mock_q_values):
            self.get_next_best_action = mock_get_action
            self.q_values = mock_q_values

    class TestHbaAgent(unittest.TestCase):
        def setUp(self):
            self.q_values_mock_test1 = unittest.mock.Mock(return_value=[])
            self.q_values_mock_test2 = unittest.mock.Mock(return_value=[])

            self.action_selection_mock_test1 = unittest.mock.Mock(
                return_value=0)
            self.action_selection_mock_test2 = unittest.mock.Mock(
                return_value=1)

            self.policy_mock_test1 = unittest.mock.Mock(return_value=0)
            self.policy_mock_test2 = unittest.mock.Mock(return_value=1)

            params = Params()
            params['max_history_items'] = 4
            params['summary']['running_average'] = 10

            self.hba_agent = HbaAgent(context=None, params=params)
            self.hba_agent.discount_factor = 1.0
            self.hba_agent.hypothesises = {
                'test1': {
                    'agent': MockTypeAgent(self.action_selection_mock_test1, self.q_values_mock_test1),
                    'policy': self.policy_mock_test1
                },
                'test2': {
                    'agent': MockTypeAgent(self.action_selection_mock_test2, self.q_values_mock_test2),
                    'policy': self.policy_mock_test2
                }
            }
            self.hba_agent._add_to_summary = lambda *args, **kwargs: None
            self.hba_agent.hypo_aliases = {'test1': 'test1', 'test2': 'test2'}
            self.hba_agent.is_first_action = False
            self.hba_agent.agent_finished = {'0': False}

        def test_likelihood_is_calculated_correctly(self):
            self.hba_agent.agent_action_prob_history = {
                '0': {
                    'test1': [0.1, 0.2, 0.5],
                    'test2': [0.3, 0.5, 0.1]
                }
            }
            self.assertEqual(self.hba_agent.agent_hypo_likelihood(
                '0', 'test1'), 0.1 * 0.2 * 0.5)
            self.assertEqual(self.hba_agent.agent_hypo_likelihood(
                '0', 'test2'), 0.3 * 0.5 * 0.1)

        def test_posterior_is_calculated_correctly(self):
            fake_hypo_prob = {
                '0': {
                    'test1': 0.25,
                    'test2': 0.5
                }
            }
            self.hba_agent.last_posterior = {
                '0': {
                    'test1': 1,
                    'test2': 1
                }
            }
            self.hba_agent.prior = {
                'test1': 0.5,
                'test2': 0.5
            }
            self.hba_agent.agent_hypo_likelihood = lambda x, y: fake_hypo_prob[x][y]
            self.assertEqual(
                self.hba_agent.agent_posterior('0', 'test1'), 0.25 / 0.75)
            self.assertEqual(
                self.hba_agent.agent_posterior('0', 'test2'), 0.5 / 0.75)

        def test_hypo_prob_is_calculated_correctly(self):
            fake_likelihood = {
                '0': {
                    'test1': 0.25,
                    'test2': 0.75
                }
            }
            self.hba_agent.agent_hypo_likelihood = lambda x, y: fake_likelihood[x][y]
            self.assertEqual(self.hba_agent.agent_hypo_likelihood('0', 'test1'), 0.25)
            self.assertEqual(self.hba_agent.agent_hypo_likelihood('0', 'test2'), 0.75)

        def test_policy_is_stored_correctly(self):
            pass

        def test_policy_is_saved_correctly(self):
            pass

    unittest.main(exit=False)
