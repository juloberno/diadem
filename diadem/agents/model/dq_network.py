# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
from diadem.agents.model.q_network import QNetwork

from diadem.agents.utils import z_range


class DistributionalQNetwork(QNetwork):
    def __init__(self, layer_in, dim_in, dim_out, params):
        self.params = params
        self.v_min = params['min_reward']
        self.v_max = params['max_reward']
        self.nb_atoms = params['num_atoms']
        self.quantile_regression = params['quantile_regression']

        super().__init__(layer_in, dim_in,
                         dim_out=dim_out*self.nb_atoms, params=params)
        p_out = self.q_output

        if self.quantile_regression:
            p_out = tf.reshape(
                p_out, shape=[-1, dim_out, self.nb_atoms], name="quantiles")
            q_out = self.quant_to_q(p_out)
        else:
            p_out = tf.reshape(
                p_out, shape=[-1, dim_out, self.nb_atoms])
            p_out = tf.nn.softmax(p_out, dim=-1, name='softmax')
            q_out = self.p_to_q(p_out)

        self.p_output = p_out
        self.q_output = q_out

    @property
    def p(self):
        """
        Get probabilities for receiving a specific reward (splitted in atoms) for a specific action in a batch

        Returns
        -------
        p : `numpy.ndarray` (batch_size, num_actions, num_atoms)
            Predicted probabilities for receiving a specific Q-value atom for choosing an action of the inserted batch values.
        """
        return self.p_output

    def p_to_q(self, p_values):
        z, _ = z_range(self.v_min, self.v_max, self.nb_atoms)
        return tf.tensordot(p_values, z, [[-1], [-1]])

    def quant_to_q(self, quantiles):
        return tf.reduce_mean( quantiles, axis=-1)
