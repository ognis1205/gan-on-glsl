import tensorflow as tf


def conv2d(
    input: tf.Tensor,
    output_dim: int,
    k_h: int = 5,
    k_w: int = 5,
    d_h: int = 2,
    d_w: int = 2,
    stddev: float = 0.02,
    name: str = 'conv2d'
) -> tf.Tensor:
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w',
            [k_h, k_w, input.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            input,
            w,
            strides=[1, d_h, d_w, 1],
            padding='SAME')
        biases = tf.get_variable(
            'biases',
            [output_dim],
            initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(
            tf.nn.bias_add(conv, biases),
            conv.get_shape())
        return conv
