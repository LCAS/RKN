import tensorflow as tf
from tensorflow import keras as k


class LayerNormalization(k.layers.Layer):
    """rudimentary implementation of layer normalization (Ba et al, 2016) as a keras layer -
    should become obsolete with tensorflow 2.0 (which provides keras.layers.experimental.LayerNormalization)"""

    def build(self, input_shape):
        shape = input_shape[-1:]
        self._offset = self.add_weight("offset", shape=shape, initializer=k.initializers.zeros)
        self._scale = self.add_weight("weight", shape=shape, initializer=k.initializers.ones)

    def call(self, inputs, **kwargs):
        """
        normalizes input over all axis but first 1 (i.e. "batch axis")
        :param inputs:
        :param kwargs:
        :return:
        """
        norm_axis = [i + 1 for i in range(len(inputs.get_shape()) - 1)]
        mean, var = tf.nn.moments(inputs, axes=norm_axis, keep_dims=True)
        return tf.nn.batch_normalization(inputs,
                                         mean=mean, variance=var, variance_epsilon=1e-12,
                                         offset=self._offset, scale=self._scale)

    def compute_output_shape(self, input_shape):
        return input_shape
