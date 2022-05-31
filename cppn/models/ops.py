import math
import numpy as np
import tensorflow as tf
from typing import Optional


def linear(
    input: tf.Tensor,
    output_size: int,
    scope: Optional[str] = None,
    stddev: float = 1.0,
    bias_start float = 0.0,
    verbose: bool = False
) -> tf.Tensor:
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable(
            "matrix",
            [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(
            "bias",
            [output_size],
            initializer=tf.constant_initializer(bias_start))
        if verbose:
            return tf.matmul(input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input, matrix) + bias


def fully_connected(
    input: tf.Tensor,
    output_size: int,
    scope: Optional[str] = None,
    stddev: float = 1.0,
    with_bias: bool = True
) -> tf.Tensor:
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or "fully_connected"):
        matrix = tf.get_variable(
            "matrix",
            [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        result = tf.matmul(input, matrix)
        if with_bias:
            bias = tf.get_variable(
                "bias",
                [1, output_size],
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias*tf.ones([shape[0], 1], dtype=tf.float32)
        return result
