# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
import numpy as np


def huber_loss(x, params, numpy=False):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""

    delta = params['delta']

    func = _huber_loss_np if numpy else _huber_loss_tf
    return func(x, delta)


def _huber_loss_tf(x, delta):
    with tf.variable_scope('huber_loss'):
        x_abs = tf.abs(x)
        return tf.where(
            x_abs < delta,
            tf.square(x) * 0.5,
            delta * (x_abs - (0.5 * delta)))


def _huber_loss_np(x, delta):
    x_abs = np.abs(x)
    return np.where(
        x_abs < delta,
        np.square(x) * 0.5,
        delta * (x_abs - (0.5 * delta))
    )


if __name__ == '__main__':
    class TestHuberLoss(tf.test.TestCase):
        def setUp(self):
            with self.cached_session() as session:
                self.x = tf.Variable(0.0)
                self.delta = tf.Variable(0.0)
                self.op = huber_loss(self.x, {'delta': self.delta})

        def test_fist_case_tf(self):
            """
            Test case if |x| < delta -> x^2 * 0.5
            """
            with self.cached_session() as session:
                loss = session.run(self.op, feed_dict={
                    self.x: -0.2,
                    self.delta: 0.5
                })
                self.assertAlmostEqual(loss, (-0.2)**2 * 0.5, places=7)

        def test_second_case_tf(self):
            """
            Test case if |x| >= delta -> delta * (x - (0.5 * delta))
            """
            with self.cached_session() as session:
                loss = session.run(self.op, feed_dict={
                    self.x: -10.0,
                    self.delta: 0.25
                })
                self.assertAlmostEqual(
                    loss, 0.25 * (10 - (0.5 * 0.25)), places=7)

        def test_fist_case_np(self):
            """
            Test case if |x| < delta -> x^2 * 0.5
            """
            loss = huber_loss(-0.2, {'delta': 0.5}, numpy=True)
            self.assertAlmostEqual(loss, (-0.2)**2 * 0.5, places=7)

        def test_second_case_np(self):
            """
            Test case if |x| >= delta -> delta * (x - (0.5 * delta))
            """
            loss = huber_loss(-10.0, {'delta': 0.25}, numpy=True)
            self.assertAlmostEqual(
                loss, 0.25 * (10 - (0.5 * 0.25)), places=7)

    try:
        tf.test.main()
    except SystemExit:
        pass
