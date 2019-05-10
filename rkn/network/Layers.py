import tensorflow as tf

#TODO remove contrib dependencies here, change to tf.layers, handle normalization
"""Wrapping default tensorflow layers for:
    - different default values
    - more readability in graph visualization"""

def dense_layer(inputs,
                output_dim,
                name,
                is_training=None,
                activation=tf.nn.relu,
                weight_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                bias_initializer=tf.random_normal_initializer(mean=0, stddev=0.0001),
                normalize_layer=False,
                keep_prob=1.0):
    """
    dense_layer
    :param inputs: Batch to process
    :param is_training: Bool tensor indicating whether training or testing
    :param output_dim: width of layer (number of neurons)
    :param name: name of layer for better readability and identification
    :param activation: activation function
    :param weight_initializer: initializer for the weights
    :param bias_initializer: initializer for the bias
    :param normalize_layer: if true use layer normalization
    :return: output of layer
    """
    if keep_prob < 1.0 and is_training is None:
        raise AssertionError("If dropout should be used, 'is_training' needs to be passed")

    with tf.name_scope(name):
        if normalize_layer:
            layer_normal = lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
        else:
            layer_normal = None

        h1 = tf.contrib.layers.fully_connected(inputs=inputs,
                                               num_outputs=output_dim,
                                               activation_fn=activation,
                                               normalizer_fn=layer_normal,
                                               weights_initializer=weight_initializer,
                                               biases_initializer=bias_initializer)
        if keep_prob < 1.0:
            h1 = tf.contrib.layers.dropout(inputs=h1,
                                           keep_prob=keep_prob,
                                           is_training=is_training)
        return h1

def convolutional_layer(inputs,
                        out_channels,
                        name,
                        filter_size,
                        strides=1,
                        padding='same',
                        on_gpu=False,
                        activation=tf.nn.relu,
                        pool=True,
                        pool_size=2,
                        pool_strides=2,
                        filter_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                        bias_initializer=tf.zeros_initializer(),
                        normalize_layer=False):
    """Convolutional Layer

    :param inputs: Batch to process
    :param out_channels: number of output channels (filters)
    :param name: name of the layer - for better readability and identification
    :param filter size: size of each individual filter
    :param strides: stride of the filtering operation
    :param padding: type of padding for both, filtering and pooling,
    :param on_gpu: true if layers are run on gpu, triggers change of format from NHWC to NCHW (cuda default)
    :param activation function
    :param pool: whether to pool or not
    :param pool_size:
    :param pool_strides: strides of the pooling operation
    :param filter_initializer: initializer for the filter
    :param bias_initializer: initializer for the bias
    :param normalize_layer: if true layer normalization is used
    :return: output of layer
    """
    with tf.name_scope(name):
        if normalize_layer:
            layer_normal = lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
        else:
            layer_normal = None

        data_format = "NCHW" if on_gpu else "NHWC"

        out = tf.contrib.layers.conv2d(inputs=inputs,
                                       num_outputs=out_channels,
                                       kernel_size=filter_size,
                                       stride=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       activation_fn=activation,
                                       normalizer_fn=layer_normal,
                                       weights_initializer=filter_initializer,
                                       biases_initializer=bias_initializer)
        if pool:
            return tf.contrib.layers.max_pool2d(out,
                                                kernel_size=pool_size,
                                                stride=pool_strides,
                                                padding=padding,
                                                data_format=data_format)
        else:
            return out


def up_convolutional_layer(inputs,
                           out_channels,
                           name,
                           filter_size,
                           strides=2,
                           padding='same',
                           on_gpu=False,
                           activation=tf.nn.relu,
                           pool=None,
                           pool_size=-1,
                           pool_strides=-1,
                           filter_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                           bias_initializer=tf.zeros_initializer(),
                           normalize_layer=False):
    """
    Up (Transposed) Convolutional layer
    :param inputs: Batch to process
    :param out_channels: number of output channels (filters)
    :param name: name of the layer - for better readability and identification
    :param filter size: size of each individual filter
    :param strides: stride of the filtering operation
    :param on_gpu: true if layers are run on gpu, triggers change of format from NHWC to NCHW (cuda default)
    :param padding: type of padding for both, filtering and pooling,
    :param activation function
    :param pool_function: currently supports tf.layers.max_pooling2d and tf.layers.average_pooling2d
    :param pool_size:
    :param pool_strides: strides of the pooling operation
    :param filter_initializer: initializer for the filter
    :param bias_initializer: initializer for the bias
    :param normalize_layer: if true layer normalization is used
    :return: output of layer
    """
    with tf.name_scope(name):
        if pool_strides != -1 or pool_size != -1 or pool is not None:
            tf.logging.warn("Pooling variables ignored for upconvolution!")

        if normalize_layer:
            layer_normal = lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
        else:
            layer_normal = None

        data_format = "NCHW" if on_gpu else "NHWC"

        return tf.contrib.layers.conv2d_transpose(inputs=inputs,
                                                  num_outputs=out_channels,
                                                  kernel_size=filter_size,
                                                  stride=strides,
                                                  padding=padding,
                                                  data_format=data_format,
                                                  activation_fn=activation,
                                                  normalizer_fn=layer_normal,
                                                  weights_initializer=filter_initializer,
                                                  biases_initializer=bias_initializer)