import tensorflow as tf


def lrelu(
    x: tf.Tensor,
    leak: float = 0.2,
    name: str = 'lrelu'
) -> tf.Tensor:
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
