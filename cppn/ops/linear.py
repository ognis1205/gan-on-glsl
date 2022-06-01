import tensorflow as tf
from typing import Optional


def linear(
    input: tf.Tensor,
    output_size: int,
    scope: Optional[str] = None,
    stddev: float = 0.02,
    bias_start: float = 0.0,
    verbose: bool = False
) -> tf.Tensor:
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or 'linear'):
        matrix = tf.get_variable(
            'matrix',
            [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(
            'bias',
            [output_size],
            initializer=tf.constant_initializer(bias_start))
        if verbose:
            return tf.matmul(input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input, matrix) + bias
