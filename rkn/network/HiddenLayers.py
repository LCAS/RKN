from abc import ABC, abstractmethod

import tensorflow as tf

from network.Layers import dense_layer, convolutional_layer, up_convolutional_layer

"""Various Classes allowing for easy initialization of multiple hidden layers with same architecture"""


class HiddenLayersParamsKeys:
    """Keys to specify network architecture - not all keys needed for each network"""
    """ All Networks, needed once per network """
    NUM_LAYERS = 'num_layers'
    ACTIVATION_FN = 'activation_fn'
    LAYER_NORMALIZATION = 'layer_normalization'
    KEEP_PROB_PREFIX = 'keep_prob_'
    """ Only dense networks, needed once per layer """
    WIDTH_PREFIX = 'width_'
    """ Only convolutional networks, needed once per network: """
    POOL = 'pool'
    POOL_FN = 'pooling_fn' # either tf.layers.max_pooling2d or tf.layers.average_pooling2d
    PADDING = 'padding' # either 'same' or 'valid'
    """ Only convolutional networks, needed once per layers"""
    FILTER_SIZE_PREFIX = 'filter_size_'
    OUT_CHANNELS_PREFIX = 'out_channels_'
    STRIDES_PREFIX = 'strides_'
    POOL_SIZE_PREFIX = 'pool_size_'
    POOL_STRIDE_PREFIX = 'pool_strides_'
    """other"""
    INIT_SHAPE = 'init_shape'

class HiddenLayers(ABC):
    """ HiddenLayers: Abstract Base Class for Hidden Layers  """

    def __init__(self, params, name_prefix):
        """
        :param params: dict - uses HiddenLayersParamsKeys to specify architecture
        :param name_prefix: name_prefix of the submodel
        """
        self.params = params
        self.name_prefix = name_prefix
        self.num_of_layers = params[HiddenLayersParamsKeys.NUM_LAYERS]
        if HiddenLayersParamsKeys.LAYER_NORMALIZATION in self.params:
            self.normalize_layer = params[HiddenLayersParamsKeys.LAYER_NORMALIZATION]
        else:
            self.normalize_layer = False

        if HiddenLayersParamsKeys.ACTIVATION_FN in self.params:
            self.activation = params[HiddenLayersParamsKeys.ACTIVATION_FN]
        else:
            self.activation = tf.nn.relu

    def __call__(self, inputs, is_training):
        """ propagates a batch of inputs through the hidden layers
        :param inputs: the inputs to process
        :return: values after last hidden layer
        """
        current = inputs
        for i in range(self.num_of_layers):
            current = self._propagate_through_layer(current, is_training, i)
        return current

    @abstractmethod
    def _propagate_through_layer(self, input, is_training, i):
        """ propagates the batch of inputs through hidden layer i
        :param input: input (current hidden activations)
        :param i: number of layer to propagate through
        :return: next hidden activations
        """
        raise NotImplementedError('Not Implemented')


class NDenseHiddenLayers(HiddenLayers):
    """ N Fully Connected (Dense) hidden layers"""

    def __init__(self, params, name_prefix):
        """see super"""
        super().__init__(params, name_prefix)

    def _propagate_through_layer(self, inputs, is_training, i):
        """see super"""
        layer_name = self.name_prefix + '_dense' + str(i + 1)
        out_dim = self.params[HiddenLayersParamsKeys.WIDTH_PREFIX + str(i + 1)]

        keep_prob_key = HiddenLayersParamsKeys.KEEP_PROB_PREFIX + str(i + 1)
        keep_prob = self.params[keep_prob_key] if keep_prob_key in self.params else 1.0

        return dense_layer(inputs=inputs,
                           is_training=is_training,
                           output_dim=out_dim,
                           name=layer_name,
                           activation=self.activation,
                           normalize_layer=self.normalize_layer,
                           keep_prob=keep_prob)


class NConvolutionalHiddenLayers(HiddenLayers):
    """ N Convolutional layers (with max pooling) """

    def __init__(self, params, name_prefix, flatten_output, on_gpu, up_convolutional):
        """
        :param params: see super
        :param name_prefix: see super
        :param flatten_output: if true (default) output filter responses of last layer are flattened into vector
        :param on_gpu: true if layers are run on gpu, triggers change of format from NHWC to NCHW (cuda default)
        """

        self.on_gpu = on_gpu

        super().__init__(params, name_prefix)
        self.flatten_output = flatten_output

        if HiddenLayersParamsKeys.POOL in self.params:
            self.pool = self.params[HiddenLayersParamsKeys.POOL]
        else:
            self.pool = True

        if HiddenLayersParamsKeys.PADDING in self.params:
            self.padding = self.params[HiddenLayersParamsKeys.PADDING]
        else:
            self.padding = 'same'

        if HiddenLayersParamsKeys.POOL_FN in self.params:
            self.pool_fn = self.params[HiddenLayersParamsKeys.POOL_FN]
        else:
            self.pool_fn = tf.layers.max_pooling2d

        self.layer = up_convolutional_layer if up_convolutional else convolutional_layer
        self.up_convolutional = up_convolutional

    def __call__(self, inputs, is_training):
        """see super"""
        res = super().__call__(inputs, is_training)

        if self.flatten_output:
            return tf.contrib.layers.flatten(res)
        else:
            return res

    def _propagate_through_layer(self, inputs, is_training, i):
        """see super"""
        filter_size = self.params[HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + str(i + 1)]
        out_channels = self.params[HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + str(i + 1)]
        strides = self.params[HiddenLayersParamsKeys.STRIDES_PREFIX + str(i + 1)]

        full_pool_size_key = HiddenLayersParamsKeys.POOL_SIZE_PREFIX + str(i + 1)
        pool_size = self.params[full_pool_size_key] if full_pool_size_key in self.params else -1

        full_pool_stride_key = HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + str(i + 1)
        pool_strides = self.params[full_pool_stride_key] if full_pool_stride_key in self.params else -1

        layer_name = self.name_prefix + ("_up" if self.up_convolutional else "") + "_convolutional" + str(i+1)

        if HiddenLayersParamsKeys.KEEP_PROB_PREFIX + str(i + 1) in self.params:
            raise AssertionError("Dropout for convolutional layers not yet implemented")

        return self.layer(inputs=inputs,
                          out_channels=out_channels,
                          name=layer_name,
                          on_gpu=self.on_gpu,
                          filter_size=filter_size,
                          pool=self.pool,
                          strides=strides,
                          padding=self.padding,
                          activation=self.activation,
                          pool_size=pool_size,
                          pool_strides=pool_strides,
                          normalize_layer=self.normalize_layer)


class HiddenLayerWrapper(HiddenLayers):
    """Used to concatenate several HiddenLayer structures (e.g dense after convolution) """

    def __init__(self, hidden_layer_list):
        """
        :param hidden_layer_list: List of hidden layers - the inputs are propagated through all hidden layers in the
                                                          order they are placed in the list.
        """
        self.hidden_layer_list = hidden_layer_list

    def __call__(self, inputs, is_training):
        """
        :param inputs: batch of inputs
        :return: output of last hidden layer of last element of hidden_layer_list
        """
        current = inputs
        for layers in self.hidden_layer_list:
            current = layers(current, is_training)
        return current

    def _propagate_through_layer(self, input, is_training, i):
        raise AssertionError('Should not be called!')

class ReshapeLayer(HiddenLayers):
    """Used to reshape tensors e.g. at the transition from dense to transposed convolutional layers"""
    def __init__(self, shape, name_prefix, on_gpu):
        """see super"""
        params = {HiddenLayersParamsKeys.NUM_LAYERS: 1}
        super().__init__(params, name_prefix)
        if on_gpu:
            self.shape = [shape[2], shape[0], shape[1]]
        else:
            self.shape = shape

    def _propagate_through_layer(self, input, is_training, i):
        with tf.name_scope(self.name_prefix + "_reshape"):
            return tf.reshape(input, tf.constant([-1] + self.shape, dtype=tf.int32))