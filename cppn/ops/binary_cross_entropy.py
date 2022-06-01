import tensorflow as tf
from typing import Optional
from tensorflow.python.framework import ops


def binary_cross_entropy(
    logits: tf.Tensor,
    targets: tf.Tensor,
    name: Optional[str] = None
) -> tf.Tensor:
    eps = 1e-12
    with ops.op_scope([logits, targets], name, 'bce_loss') as name:
        ls = ops.convert_to_tensor(logits, name='logits')
        ts = ops.convert_to_tensor(targets, name='targets')
        return tf.reduce_mean(
            -(ls * tf.log(ts + eps) + (1. - ls) * tf.log(1. - ts + eps)))
