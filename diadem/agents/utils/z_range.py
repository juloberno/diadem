# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import numpy as np
import tensorflow as tf


def z_range(min_value, max_value, num_items, numpy=False):
    """
    Creates an interval from [min_value, max_value] with the interval of num_items

    Creates a tuple of an array with including min and max values in the array as first argument and the
    array step to the next array element as second tuple element. It will have because of this the
    following format.
    If numpy is false, a tensorflow operation is returned. If not, it will return an numpy array.

    Example:
    >>> z_range(0, 10, 10, numpy=True)
    array([ 0.        ,  1.11111111,  2.22222222,  3.33333333,  4.44444444,
        5.55555556,  6.66666667,  7.77777778,  8.88888889, 10.        ]), 1.111111111
    """
    step = (max_value - min_value) / (num_items - 1)
    if numpy:
        interval = np.arange(min_value, max_value + step / 2, step)
    else:
        interval = tf.range(min_value, max_value + step / 2,
                            step, dtype=tf.float32, name='z')

    return interval, step


if __name__ == '__main__':
    class TestZRange(tf.test.TestCase):
        min_value = 0
        max_value = 10
        num_items = 10
        output = (
            [
                0., 1.11111111, 2.22222222, 3.33333333, 4.44444444,
                5.55555556, 6.66666667, 7.77777778, 8.88888889, 10.
            ],
            1.111111111
        )

        def test_tf_implementation(self):
            with self.cached_session() as session:
                min_value = tf.constant(self.min_value, dtype=tf.float32)
                max_value = tf.constant(self.max_value, dtype=tf.float32)
                num_items = tf.constant(self.num_items, dtype=tf.float32)
                op = _z_range(min_value, max_value, num_items)

                interval, step = session.run(op)

                self.assertArrayNear(interval, self.output[0], err=1e-6)
                self.assertAlmostEqual(step, self.output[1], places=5)

        def test_numpy_implementation(self):
            interval, step = _z_range(
                self.min_value, self.max_value, self.num_items, numpy=True)

            self.assertArrayNear(list(interval), self.output[0], err=1e-6)
            self.assertAlmostEqual(step, self.output[1], places=5)

    try:
        tf.test.main()
    except SystemExit:
        pass
