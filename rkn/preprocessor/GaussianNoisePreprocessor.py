import tensorflow as tf

class GaussianNoisePreprocessor:

    def __init__(self, std):

        self._std = std

    def __call__(self, observations, on_gpu):
        noise = tf.random_normal(tf.shape(observations), mean=0, stddev=self._std, dtype=observations.dtype)
        return observations + noise
