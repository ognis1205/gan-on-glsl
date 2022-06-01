import tensorflow as tf
from typing import Optional


def fully_connected(
    input: tf.Tensor,
    output_size: int,
    scope: Optional[str] = None,
    stddev: float = 0.1,
    verbose: bool = True
) -> tf.Tensor:
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or 'fully_connected'):
        matrix = tf.get_variable(
            'matrix',
            [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        result = tf.matmul(input, matrix)
        if verbose:
            bias = tf.get_variable(
                'bias',
                [1, output_size],
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias * tf.ones([shape[0], 1], dtype=tf.float32)
        return result
