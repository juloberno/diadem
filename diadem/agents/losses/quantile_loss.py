# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
import numpy as np
from diadem.agents.model import DistributionalQNetwork

from diadem.agents.losses.categorical_loss import numpy_gather_nd, batch_action_pair
from diadem.agents.losses.huber_loss import huber_loss


def quantile_loss(estimation_network: DistributionalQNetwork, target_network: DistributionalQNetwork, actions, evaluation_actions, rewards, done_mask, params, numpy=False):
    """
    Calculates the quantile loss for a given distributed target and estimation network

    This method uses the distribution (estimation_network.p and target_network.p) to calculate the
    distributional quantile loss for the next actions.

    For more information see https://arxiv.org/pdf/1710.10044.pdf

    Parameters
    ----------
    estimation_network : `DistributionalQNetwork`
        Estimation network that is used for predicting Q(s, a)
    target_network : `DistributionalQNetwork`
        Target network to evaluate the estimations of the estimation network
    actions: `numpy.ndarray`, (batch_size)
        Array of predicted actions of every batch
    evaluation_actions: `numpy.ndarray`, (batch_size)
        Array of actions that are used for evaluating the predicted actions.
    rewards:  `numpy.ndarray`, (batch_size)
        Array of rewards that were received from the environment for every batch item
    done_mask: `numpy.ndarray`, (batch_size)
        Array of boolean values that indicate if batch run is done (~1) or not (~0)
    params: `dict`
        Parameters to configure the quantile loss

        - ``huber_loss``: `dict`
            Parameters for huber loss calculation

            - ``delta``: delta Parameter for huber loss between 0 and 1. Please use 1.0 as default parameter (`float32`)
        - ``discount_factor``: discount factor with which the target prediction is multiplied
    """
    estimation_network_p = estimation_network.p
    target_network_p = target_network.p

    # parameters
    discount_factor = params['discount_factor']
    batch_size, num_actions, num_atoms = estimation_network_p.shape
    batch_size = params['batch_size']
    batch_size_dim, num_actions_dim, num_atoms_dim = estimation_network_p.shape
    num_actions = int(num_actions_dim)
    num_atoms = int(num_atoms_dim)

    # functions to choose between numpy and tensorflow
    abs_f = np.abs if numpy else tf.abs
    reduce_mean = np.mean if numpy else tf.reduce_mean
    reduce_sum = np.sum if numpy else tf.reduce_sum
    gather_nd = numpy_gather_nd if numpy else tf.gather_nd

    # check for correct shape of inserted data
    assert list(target_network_p.shape[1:]) == [num_actions_dim, num_atoms_dim], 'Target network p-value requires shape %s but has %s' % (
        estimation_network_p.shape, target_network_p.shape)
    assert len(rewards.shape) is 1, 'Rewards requires shape %s but has %s' % (
        (batch_size_dim), list(rewards.shape))
    assert len(actions.shape) is 1, 'Actions requires shape %s but has %s' % (
        (batch_size_dim), list(actions.shape))
    assert len(evaluation_actions.shape) is 1, 'Evaluation actions requires shape %s but has %s' % (
        (batch_size_dim), list(evaluation_actions.shape))
    assert len(done_mask.shape) is 1, 'Done mask requires shape %s but has %s' % (
        (batch_size_dim), list(done_mask.shape))

    # check for correct types
    if not numpy:
        tf.assert_type(estimation_network_p, tf.float32, 'Estimation network p-value requires type float32 but has %s' % (
            estimation_network_p.dtype))
        tf.assert_type(target_network_p, tf.float32, 'Target network p-value requires type float32 but has %s' % (
            target_network_p.dtype))
        tf.assert_type(rewards, tf.float32, 'Rewards requires type float32 but has %s' % (
            rewards.dtype))
        tf.assert_type(actions, tf.int32, 'Actions requires type int32 but has %s' % (
            actions.dtype))
        tf.assert_type(evaluation_actions, tf.int32, 'Evaluation actions requires type int32 but has %s' % (
            evaluation_actions.dtype))
        tf.assert_type(done_mask, tf.float32, 'Done mask requires type float32 but has %s' % (
            done_mask.dtype))

    def _batch_action_pair(actions):
        return batch_action_pair(batch_size, actions, numpy=numpy)

    def _batch_action_probabilities(probs, action_batches):
        idx = _batch_action_pair(action_batches)
        return gather_nd(probs, idx)

    def _huber_loss(error):
        return huber_loss(error, params['huber_loss'], numpy=numpy)

    # quantiles for actions which we know were selected in the given state x.
    quant_t_selected = _batch_action_probabilities(
        estimation_network_p, actions)

    # target quantiles for actions we predicted for the next state x'
    quant_t_next = _batch_action_probabilities(
        target_network_p, evaluation_actions)

    # mask all already finished batches
    quant_t_next_unfinished = quant_t_next * (1 - done_mask[:, None])

    # calculate target_reward = r + gamma * max Q_target(s', a')
    quant_target = rewards[:, None] + \
        discount_factor * quant_t_next_unfinished

    # calculate error with L = r + (target_reward - current_reward)
    _quant_target = quant_target[:, :, None] if numpy else tf.stop_gradient(
        quant_target[:, :, None])
    error = _quant_target - quant_t_selected[:, None, :]

    # prepare parameters for huber loss (see equation (10))
    negative_indicator = (error < 0).astype(
        np.float32) if numpy else tf.cast(error < 0, tf.float32)
    tau = np.array(range(0, num_atoms + 1)) / num_atoms
    tau_hat = (tau[:-1] + tau[1:]) / 2
    if not numpy:
        tau_hat = tf.constant(tau_hat, dtype=tf.float32, name='tau_hat')

    # calculate final loss
    _huber_loss = _huber_loss(error)
    quant_weights = abs_f(tau_hat - negative_indicator)
    _quantile_loss = quant_weights * _huber_loss
    errors = reduce_sum(reduce_mean(_quantile_loss, axis=-2), axis=-1)

    return errors


if __name__ == '__main__':
    def _old_quantile_loss(p_values, p_target, rewards, a_next, done_mask, actions, params):
        """
        Builds the parametrised quantile regression Algorithm 1 of
        'Distributional Reinforcement Learning with Quantile Regression' - http://arxiv.org/abs/1710.10044
        """

        nb_atoms = params['num_atoms']
        discount_factor = params['discount_factor']

        def _gather_along_second_axis(data, indices):
            batch_offset = tf.range(0, tf.shape(data)[0])
            flat_indices = tf.stack([batch_offset, indices], axis=1)
            return tf.gather_nd(data, flat_indices)

        def _huber_loss(error):
            return huber_loss(error, params['huber_loss'])

        # quantiles for actions which we know were selected in the given state.
        quant_t_selected = _gather_along_second_axis(p_values, actions)
        quant_t_selected.set_shape([None, nb_atoms])

        # pick next action and apply mask
        quant_tp1_star = _gather_along_second_axis(p_target, a_next)
        quant_tp1_star.set_shape([None, nb_atoms])
        quant_tp1_star = tf.einsum('ij,i->ij', quant_tp1_star, 1. - done_mask)

        # Tth = r + gamma * th
        batch_dim = tf.shape(rewards)[0]
        quant_target = tf.identity(
            rewards[:, tf.newaxis] + discount_factor * quant_tp1_star, name='quant_target')

        # increase dimensions (?, n, n)
        big_quant_target = tf.transpose(tf.reshape(tf.tile(quant_target, [1, nb_atoms]), [
            batch_dim, nb_atoms, nb_atoms], name='big_quant_target'), perm=[0, 2, 1])
        # big_quant_target[0] =
        #  [[Tth1 Tth1 ... Tth1]
        #   [Tth2 Tth2 ... Tth2]
        #   [...               ]
        #   [Tthn Tthn ... Tthn]]

        big_quant_t_selected = tf.reshape(tf.tile(quant_t_selected, [1, nb_atoms]), [
            batch_dim, nb_atoms, nb_atoms], name='big_quant_t_selected')
        # big_quant_t_selected[0] =
        #  [[th1 th2 ... thn]
        #   [th1 th2 ... thn]
        #   [...            ]
        #   [th1 th2 ... thn]]

        # build loss
        td_error = tf.stop_gradient(big_quant_target) - big_quant_t_selected
        # td_error[0]=
        #  [[Tth1-th1 Tth1-th2 ... Tth1-thn]
        #   [Tth2-th1 Tth2-th2 ... Tth2-thn]
        #   [...                           ]
        #   [Tthn-th1 Tthn-th2 ... Tthn-thn]]
        # TODO: skip tiling

        negative_indicator = tf.cast(td_error < 0, tf.float32)

        tau = tf.range(0, nb_atoms + 1, dtype=tf.float32,
                       name='tau') * 1. / nb_atoms
        tau_hat = tf.identity((tau[:-1] + tau[1:]) / 2, name='tau_hat')

        _huber_loss = _huber_loss(td_error)
        quant_weights = tf.abs(tau_hat - negative_indicator)
        _quantile_loss = quant_weights * _huber_loss

        errors = tf.reduce_sum(tf.reduce_mean(
            _quantile_loss, axis=-2), axis=-1)  # E_j # atoms

        return errors

    class MockedDQNetwork(DistributionalQNetwork):
        def __init__(self, mocked_p):
            self.mocked_p = mocked_p

        @property
        def p(self):
            return self.mocked_p

    class MockedDqNetworkTf(DistributionalQNetwork):
        def __init__(self, network):
            self.network = network

        @property
        def p(self):
            p = self.network.p
            return tf.Variable(
                p, dtype=tf.float32
            )

    class TestQuantileLoss(tf.test.TestCase):
        params2 = {
            'num_atoms': 2,  # only for only algorithm required
            'batch_size': 2,
            'huber_loss': {
                'delta': 1.0
            },
            'discount_factor': 0.95
        }

        current_actions2 = np.array([0, 1], dtype=np.int32)
        next_actions2 = np.array([1, 0], dtype=np.int32)
        rewards2 = np.array([0, 1], dtype=np.float32)
        done_mask2 = np.array([0, 1], dtype=np.float32)
        estimation_network2 = MockedDQNetwork(
            np.array([
                [
                    [0.5, 0.5],
                    [0.2, 0.8],
                    [0.5, 0.5]
                ],
                [
                    [0.9, 0.1],
                    [0.5, 0.4],
                    [0.5, 0.5]
                ]
            ], dtype=np.float32)
        )
        target_network2 = MockedDQNetwork(
            np.array([
                [
                    [0.4, 0.6],
                    [0.5, 0.5],
                    [0.5, 0.5]
                ], [
                    [0.9, 0.1],
                    [0.5, 0.4],
                    [0.5, 0.5]
                ]
            ], dtype=np.float32)
        )

        params = {
            'num_atoms': 3,  # only for only algorithm required
            'batch_size': 1,
            'huber_loss': {
                'delta': 1.0
            },
            'discount_factor': 0.95
        }

        current_actions = np.array([0], dtype=np.int32)
        next_actions = np.array([0], dtype=np.int32)
        rewards = np.array([0.5], dtype=np.float32)
        done_mask = np.array([0], dtype=np.float32)
        estimation_network = MockedDQNetwork(
            np.array([
                [
                    [0.5, 0.1, 0.4]
                ]
            ], dtype=np.float32)
        )
        target_network = MockedDQNetwork(
            np.array([
                [
                    [0.4, 0.1, 0.5],
                ]
            ], dtype=np.float32)
        )

        def _old_implementation(self, p_values, p_target, done_mask, rewards,
                                next_actions, actions, params):
            with self.cached_session() as session:
                op = _old_quantile_loss(
                    p_values=tf.Variable(p_values, dtype=tf.float32),
                    p_target=tf.Variable(p_target, dtype=tf.float32),
                    done_mask=tf.Variable(done_mask, dtype=tf.float32),
                    rewards=tf.Variable(rewards, dtype=tf.float32),
                    a_next=tf.Variable(next_actions, dtype=tf.int32),
                    actions=tf.Variable(actions, dtype=tf.int32),
                    params=params
                )
                session.run(tf.global_variables_initializer())
                return session.run(op)
        # actions = current_actions
        # evaluation_actions = next_actions = 
        def test_numpy_implementation_with_1batch(self):
            loss = quantile_loss(
                rewards=self.rewards,
                done_mask=self.done_mask,
                actions=self.current_actions,
                evaluation_actions=self.next_actions,
                estimation_network=self.estimation_network,
                target_network=self.target_network,
                params=self.params,
                numpy=True
            )
            should_value = self._old_implementation(
                self.estimation_network.p,
                self.target_network.p,
                self.done_mask,
                self.rewards,
                self.next_actions,
                self.current_actions,
                self.params
            )
            self.assertArrayNear(should_value, loss, 1e-4)

        def test_numpy_implementation_with_2batches(self):
            loss = quantile_loss(
                rewards=self.rewards2,
                done_mask=self.done_mask2,
                actions=self.current_actions2,
                evaluation_actions=self.next_actions2,
                estimation_network=self.estimation_network2,
                target_network=self.target_network2,
                params=self.params2,
                numpy=True
            )
            should_value = self._old_implementation(
                self.estimation_network2.p,
                self.target_network2.p,
                self.done_mask2,
                self.rewards2,
                self.next_actions2,
                self.current_actions2,
                self.params2
            )
            self.assertArrayNear(should_value, loss, 1e-4)

        def test_tf_implementation_with_1batch(self):
            loss = quantile_loss(
                rewards=tf.Variable(self.rewards, dtype=tf.float32),
                done_mask=tf.Variable(self.done_mask, dtype=tf.float32),
                actions=tf.Variable(
                    self.current_actions, dtype=tf.int32),
                evaluation_actions=tf.Variable(
                    self.next_actions, dtype=tf.int32),
                estimation_network=MockedDqNetworkTf(self.estimation_network),
                target_network=MockedDqNetworkTf(self.target_network),
                params=self.params,
                numpy=False
            )
            should_value = self._old_implementation(
                self.estimation_network.p,
                self.target_network.p,
                self.done_mask,
                self.rewards,
                self.next_actions,
                self.current_actions,
                self.params
            )
            with self.cached_session() as session:
                session.run(tf.global_variables_initializer())
                loss = session.run(loss)
                self.assertArrayNear(should_value, loss, 1e-4)

        def test_tf_implementation_with_2batches(self):
            loss = quantile_loss(
                rewards=tf.Variable(self.rewards2, dtype=tf.float32),
                done_mask=tf.Variable(self.done_mask2, dtype=tf.float32),
                actions=tf.Variable(
                    self.current_actions2, dtype=tf.int32),
                evaluation_actions=tf.Variable(
                    self.next_actions2, dtype=tf.int32),
                estimation_network=MockedDqNetworkTf(self.estimation_network2),
                target_network=MockedDqNetworkTf(self.target_network2),
                params=self.params2,
                numpy=False
            )
            should_value = self._old_implementation(
                self.estimation_network2.p,
                self.target_network2.p,
                self.done_mask2,
                self.rewards2,
                self.next_actions2,
                self.current_actions2,
                self.params2
            )
            with self.cached_session() as session:
                session.run(tf.global_variables_initializer())
                loss = session.run(loss)
                self.assertArrayNear(should_value, loss, 1e-4)


if __name__ == '__main__':
    try:
        tf.test.main()
    except SystemExit:
        pass
