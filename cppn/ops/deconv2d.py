import tensorflow as tf
from typing import Sequence


def deconv2d(
    input: tf.Tensor,
    output_shape: Sequence,
    k_h: int = 5,
    k_w: int = 5,
    d_h: int = 2,
    d_w: int = 2,
    stddev: float = 0.02,
    name: str = 'deconv2d',
    verbose: bool = False
) -> tf.Tensor:
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w',
            [k_h, k_h, output_shape[-1], input.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.deconv2d(
            input,
            w,
            output_shape=output_shape,
            strides=[1, d_h, d_w, 1])
        biases = tf.get_variable(
            'biases',
            [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(
            tf.nn.bias_add(deconv, biases),
            deconv.get_shape())
        if verbose:
            return deconv, w, biases
        else:
            return deconv
