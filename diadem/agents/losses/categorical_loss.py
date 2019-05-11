# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import numpy as np
import tensorflow as tf

from diadem.agents.model import DistributionalQNetwork
from diadem.agents.utils import z_range


def numpy_gather_nd(arr, indexes):
    return np.array(
        [arr[batch_idx[0]][batch_idx[1]] for batch_idx in indexes],
        dtype=arr.dtype
    )


if __name__ == '__main__':
    class TestNumpyGatherNd(tf.test.TestCase):
        def test_with_3d_example(self):
            idx = [[0, 1], [0, 0]]
            arr = np.array([[[0.5, 0.5], [0.2, 0.8]]])
            res = list(numpy_gather_nd(arr, idx))
            target = np.array([[0.2, 0.8], [0.5, 0.5]])
            self.assertAllEqual(
                res,
                target
            )

        def test_has_same_behavior_as_tf(self):
            idx = [[0, 1], [0, 0]]
            arr = np.array([[[0.5, 0.5], [0.2, 0.8]]])
            res = numpy_gather_nd(arr, idx)
            with self.cached_session() as session:
                op = tf.gather_nd(arr, idx)
                session.run(tf.global_variables_initializer())
                res_tf = session.run(op)

            self.assertAllEqual(
                res,
                res_tf
            )


def batch_action_pair(batch_size, actions, numpy=False):
    transpose = np.transpose if numpy else tf.transpose
    reshape = np.reshape if numpy else tf.reshape
    range_f = range if numpy else tf.range
    return transpose(reshape([range_f(batch_size), actions], [2, batch_size]))


def categorical_loss(estimation_network: DistributionalQNetwork, target_network: DistributionalQNetwork, actions, evaluation_actions, rewards, done_mask, params, numpy=False):
    """
    Calculates the categorical loss for a given distributed target and estimation network

    This method uses the distribution (estimation_network.p and target_network.p) to calculate the
    distributional categorical loss for the next actions.

    For more information, see https://arxiv.org/pdf/1707.06887.pdf

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
        Array of rewards that were predicted for every batch item
    done_mask: `numpy.ndarray`, (batch_size)
        Array of boolean values that indicate if batch run is done (~1) or not (~0)
    params: `dict`
        Parameters to configure the categorical loss

        - ``min_reward``: `dict`
            Minimum value for predicted q values (will be predicted in interval [min_value, max_value])
        - ``max_reward``: `dict`
            Maximum value for predicted q values (will be predicted in interval [min_value, max_value])
        - ``discount_factor``: discount factor with which the target prediction is multiplied
    """
    estimation_network_p = estimation_network.p
    target_network_p = target_network.p

    # parameters
    discount_factor = params['discount_factor']
    min_reward = params['min_reward']
    max_reward = params['max_reward']
    batch_size = params['batch_size']
    batch_size_dim, num_actions_dim, num_atoms_dim = estimation_network_p.shape
    num_actions = int(num_actions_dim)
    num_atoms = int(num_atoms_dim)

    # functions to choose between numpy and tensorflow
    clip = np.clip if numpy else tf.clip_by_value
    abs_f = np.abs if numpy else tf.abs
    reduce_sum = np.sum if numpy else tf.reduce_sum
    log = np.log if numpy else tf.log
    einsum = np.einsum if numpy else tf.einsum
    gather_nd = numpy_gather_nd if numpy else tf.gather_nd

    reward_atoms, atom_diff = z_range(
        min_reward, max_reward, num_atoms, numpy=numpy)

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
        """
        create a key value list of batch and action:
        [(0, a1), (1, a0), (2, a0), ...] to fetch the batch actions
        later with gather_nd
        """
        return batch_action_pair(batch_size, actions, numpy=numpy)

    # add atoms to real rewards but remove all not already finished ones and add a discount factor
    non_finished_chunks = (1 - done_mask[:, None]) * reward_atoms[None, :]
    atom_rewards = rewards[:, None] + \
        discount_factor * non_finished_chunks

    # only allow rewards in the range min_reward and max_reward (~Tz)
    atom_rewards = clip(atom_rewards, min_reward, max_reward)

    # now calculate the equation (7)
    ThTz = einsum(
        'ijk,ik->ij',
        clip(
            1 - abs_f(atom_rewards[:, None, :] -
                      reward_atoms[None, :, None]) / atom_diff,
            0, 1),
        gather_nd(target_network_p, _batch_action_pair(evaluation_actions))
    )

    # calculate cross entropy loss
    cross_entropy = -1 * ThTz * \
        log(gather_nd(estimation_network_p, _batch_action_pair(actions)))
    cross_entropy_loss = reduce_sum(cross_entropy, axis=-1)

    return cross_entropy_loss


if __name__ == '__main__':
    def old_categorical_alg(p_values, p_target, rewards, a_next, done_mask, actions, params):
        """
        Builds the vectorized categorical algorithm following equation (7) of
        'A Distributional Perspective on Reinforcement Learning' - https://arxiv.org/abs/1707.06887
        """
        Vmin = params['min_reward']
        Vmax = params['max_reward']
        nb_atoms = params['num_atoms']
        discount_factor = params['discount_factor']
        batch_size = params['batch_size']

        z, dz = z_range(Vmin, Vmax, nb_atoms)
        with tf.variable_scope('categorical'):
            cat_idx = tf.transpose(tf.reshape(
                tf.concat([tf.range(batch_size), a_next], axis=0), [2, batch_size]))
            p_best = tf.gather_nd(p_target, cat_idx)

            cat_idx = tf.transpose(tf.reshape(
                tf.concat([tf.range(batch_size), actions], axis=0), [2, batch_size]))
            p_t_next = tf.gather_nd(p_values, cat_idx)

            big_z = tf.reshape(tf.tile(z, [batch_size]), [
                batch_size, nb_atoms])
            big_r = tf.transpose(tf.reshape(tf.tile(rewards, [nb_atoms]), [
                nb_atoms, batch_size]))

            Tz = tf.clip_by_value(big_r + discount_factor *
                                  tf.einsum('ij,i->ij', big_z, 1. - done_mask), Vmin, Vmax)

            big_Tz = tf.reshape(
                tf.tile(Tz, [1, nb_atoms]), [-1, nb_atoms, nb_atoms])
            big_big_z = tf.reshape(
                tf.tile(big_z, [1, nb_atoms]), [-1, nb_atoms, nb_atoms])

            Tzz = tf.abs(big_Tz - tf.transpose(big_big_z, [0, 2, 1])) / dz
            Thz = tf.clip_by_value(1 - Tzz, 0, 1)

            ThTz = tf.einsum('ijk,ik->ij', Thz, p_best)

            # compute the error (potentially clipped)
            cross_entropy = -1 * ThTz * tf.log(p_t_next)
            errors = tf.reduce_sum(cross_entropy, axis=-1)

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

    class TestCategoricalLoss(tf.test.TestCase):
        params2 = {
            'batch_size': 2,
            'min_reward': 0,
            'max_reward': 1,
            'num_atoms': 2,
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
            'batch_size': 1,
            'min_reward': 0,
            'max_reward': 1,
            'num_atoms': 3,
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
                op = old_categorical_alg(
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

        def test_numpy_implementation_with_1batch(self):
            loss = categorical_loss(
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
            loss = categorical_loss(
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
            loss = categorical_loss(
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
            loss = categorical_loss(
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
