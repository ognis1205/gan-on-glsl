import tensorflow as tf

class BatchNormalizer:
    def __init__(
        self,
        size: int,
        epsilon: float = 1e-5,
        momentun: float = 0.1,
        name: str = "batch_normalizer"
    ) -> None:
        self.epsilon = epsilon
        self.momentum = momentum
        self.size = size
        self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
        self.name = name

    def __call__(
        self,
        x: tf.Tensor,
        train: bool=True
    ) -> tf.Tensor:
        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable(
                "gamma",
                [shape[-1]],
                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable(
                "beta",
                [shape[-1]],
                initializer=tf.constant_initializer(0.))
            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])
            return tf.nn.batch_normalization(
                x,
                self.mean,
                self.variance,
                self.beta,
                self.gamma,
                self.epsilon)
