# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf


class QNetwork():
    def __init__(self, layer_in, dim_in, dim_out, params):
        neurons_per_hidden_layer = params['neurons_per_hidden_layer']
        hidden_layers = len(neurons_per_hidden_layer)
        _dim_in = dim_in
        _layer_in = layer_in
        # Add input and hidden layers
        for i in range(0, hidden_layers):
            _dim_out = neurons_per_hidden_layer[i]
            _layer_out = self._add_layer(
                _layer_in, _dim_in, _dim_out, activation=tf.nn.relu, name='dense' + str(i))
            _layer_in = _layer_out
            _dim_in = _dim_out

        # Add output layer
        self.q_output = self._add_layer(
            _layer_in, _dim_in, dim_out, name='output')

    def _add_layer(self, layer_in, dim_in, dim_out, name='layer', **kwargs):
        # Function to add new layers to the graph
        with tf.variable_scope(name):
            w = tf.get_variable(
                'weight', [dim_in, dim_out], initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                'bias', [dim_out], initializer=tf.constant_initializer(0.1))
            h = tf.matmul(layer_in, w) + b
            if 'activation' in kwargs:
                h = kwargs['activation'](h)
            return h

    @property
    def q(self):
        """
        Get current q values for all possible actions of the inserted batch values

        Returns
        -------
        q : `numpy.ndarray` (batch_size, num_actions)
            Predicted Q values for choosing an action in the inserted batch values
        """
        return self.q_output
