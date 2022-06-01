import tensorflow as tf


def cond_concat(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(
        3,
        [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])
