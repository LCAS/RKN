import abc
import tensorflow as tf
from network.Layers import dense_layer, up_convolutional_layer
from util import MeanCovarPacker as mcp
import numpy as np

"""Various Classes for different output layers"""

class OutputLayer(abc.ABC):
    """Abstract output layer, base class"""
    def __init__(self, output_dim, name_prefix):
        """ Create new output layer
        :param output_dim: dimensionality of output
        :param name_prefix: name of the layer
        """
        self._output_dim = output_dim
        self.name_prefix = name_prefix

    @abc.abstractmethod
    def __call__(self, last_hidden):
        """
        Propagates the last hidden activations through the output layer
        :param last_hidden: batch of last hidden activations
        :return: batch of outputs
        """
        raise NotImplementedError("NOT IMPLEMENTED")

    @property
    def output_dim(self):
        return self._output_dim


class SimpleOutputLayer(OutputLayer):
    """Standard dense output layer"""
    def __init__(self, output_dim, name_prefix, activation=tf.identity):
        """See super
        :param activation: Activation Function of the output layer"""
        super().__init__(output_dim=output_dim, name_prefix=name_prefix)
        self.activation = activation

    def __call__(self, last_hidden):
        """see super"""
        return dense_layer(inputs=last_hidden,
                           output_dim=self.output_dim,
                           name=self.name_prefix + '_SimpleOutputLayer',
                           activation=self.activation)


class UpConvOutputLayer(OutputLayer):
    """Up (transposed) convolutional output layer for image generation"""
    def __init__(self, output_dim, name_prefix, on_gpu, activation=tf.nn.sigmoid, strides=1):
        """
        see super
        :param on_gpu: true if layers are run on gpu, triggers change of format from NHWC to NCHW (cuda default)
        :param activation: Activation function
        :param strides: Stride of the transposed convolution
        """
        super().__init__(output_dim=output_dim, name_prefix=name_prefix)
        self.on_gpu = on_gpu
        self.strides=strides
        self.channels = output_dim[0] if self.on_gpu else output_dim[-1]
        self.activation = activation

    def __call__(self, last_hidden):
        """See super"""
        return up_convolutional_layer(inputs=last_hidden,
                                      filter_size=3,
                                      out_channels=self.channels,
                                      name=self.name_prefix + "_UpconvolutionOutput",
                                      strides=self.strides,
                                      on_gpu=self.on_gpu,
                                      activation=self.activation,
                                      padding='same')

class FixedOutputLayer(OutputLayer):
    """Dummy layer, does a matrix multiplication followed by a vector addition - matrix and vector fixed"""
    def __init__(self, output_dim, name_prefix, weight_matrix, bias=None):
        """
        See super
        :param weight_matrix: fixed weight matrix to be used
        :param bias: fixed bias to be used
        """
        super().__init__(output_dim=output_dim, name_prefix=name_prefix)
        self.weight_matrix = tf.constant(weight_matrix, dtype=tf.float32)
        self.bias = tf.constant(bias, dtype=tf.float32) if bias is not None else None

    def __call__(self, last_hidden):
        with tf.name_scope(self.name_prefix + "_FixedOutputLayer"):
            if self.bias is None:
                return tf.matmul(last_hidden, self.weight_matrix)
            else:
                return tf.nn.xw_plus_b(last_hidden, self.weight_matrix, self.bias)


class GaussianOutputLayer(OutputLayer):
    """ Gaussian Output Layer - Learns a Multivariate Gaussian Distribution with diagonal covariance"""

    def __init__(self, distribution_dim, name_prefix, single_variance=True, variance_fn=tf.exp, max_variance=-1,
                 constant_variance=False, fix_constant_variance=False, variance=None, normalize_mean=False):
        """
        :param distribution_dim: dimensionality of the distribution
        :param name_prefix: name for better readability and identification
        :param single_variance: whether to use a learn a single variance value or a complete vector
        :param variance_fn: which function to use to ensure the variance is positive
        :param max_variance: if this is a positive value everything covariance value above is clipped
        """

        self.distribution_dim = distribution_dim
        super().__init__(output_dim= 2 * self.distribution_dim, name_prefix=name_prefix)
        self.single_variance = single_variance
        self.variance_fn = variance_fn
        self.max_variance = max_variance

        self.constant_variance = constant_variance
        self.fix_constant_variance = fix_constant_variance
        self.variance = variance
        self._normalize_mean = normalize_mean

        if self.constant_variance:
            if self.fix_constant_variance:
                self._const_var = tf.constant(np.expand_dims(self.variance, 0), dtype=tf.float32)
                if not self.single_variance:
                    self._const_var = tf.tile(self._const_var, [1, self.distribution_dim])
            else:
                log_init = tf.tile(tf.constant([[np.log(self.variance)]]), [1, 1 if single_variance else self.distribution_dim])
                log_var = tf.get_variable(name='log_var', initializer=log_init, dtype=tf.float32)
                self._const_var = self.variance_fn(log_var)
                if self.max_variance > 0:
                    self._const_var = tf.minimum(self._const_var, self.max_variance)

    def __call__(self, inputs):
        """see super
        :return vector containing the concatenated mean and covariance
        """
        with tf.name_scope(self.name_prefix + '_GaussianOutput'):
            mean = dense_layer(inputs=inputs,
                               output_dim=self.distribution_dim,
                               name=self.name_prefix + '_mean',
                               activation=tf.identity,
                               is_training=False)
            if self._normalize_mean:
                tf.logging.warn("ENCODER CURRENTLY NORMALIZING!")
                mean = mean / tf.norm(mean, ord="euclidean", axis=-1, keepdims=True)

            if self.constant_variance:
                batch_size = tf.shape(mean)[0]
                var = tf.tile(self._const_var, [batch_size, self.distribution_dim if self.single_variance else 1])
            else:

                var = dense_layer(inputs=inputs,
                                  output_dim= 1 if self.single_variance else self.distribution_dim,
                                  name=self.name_prefix + '_logVar',
                                  activation=self.variance_fn,
                                  is_training=False)
                if self.max_variance > 0:
                    var = tf.minimum(var, self.max_variance)
                if self.single_variance:
                    var = tf.tile(var, [1, self.distribution_dim])
            return mcp.pack(mean, var)

