import tensorflow as tf
import numpy as np

class FeedForwardNet:
    """"""
    def __init__(self,
                 output_layer,
                 hidden_layers=None,
                 permute_batch_and_seq=True):
        """
        Constructs new feed forward network
        :param output_layer: output layer
        :param hidden_layers: hidden layers, if none the input is directly fed trough output layer
        :param permute_batch_and_seq: whether to permute the batch and sequence dimensions before processing, speeds up
        computation if batch size bigger than sequence lengths
        """
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.permute_batch_and_seq = False

    def __call__(self, inputs, is_training=None, sequence_data=False):
        """
        Processes batch of inputs data
        :param inputs:  batch of inputs
        :param sequence_data: indicating whether the data is of sequential data, if true shape
                [batch x time x data] is assumed
        :return: batch of outputs
        """
        if sequence_data:
            return self._propagate_sequence_data(inputs, is_training)
        else:
            return self._propagate(inputs, is_training)

    def _propagate(self, inputs, is_training):
        """Propagates batch of single data points through net
        :param inputs: batch of inputs
        :return: batch of outputs"""
        if self.hidden_layers is not None:
            last_hidden = self.hidden_layers(inputs, is_training)
        else:
            last_hidden = inputs
        return self.output_layer(last_hidden)

    def _propagate_sequence_data(self, ts_inputs, is_training):
        """ Propagates batch of time series through net
        :param ts_inputs: batch of input time series
        :return: batch of output time series
        """
        if self.permute_batch_and_seq:
            transposed_in = self._transpose_sequence(ts_inputs)
            results = tf.concat(tf.map_fn(lambda input: self._propagate(input, is_training), transposed_in), 0)
            return self._transpose_sequence(results)
        else:
            return tf.concat(tf.map_fn(lambda input: self._propagate(input, is_training), ts_inputs), 0)

    @staticmethod
    def _transpose_sequence(tensor):
        """Changes batch and time dimension
        :param tensor: input tensor
        :return: tensor with transposed dimensions
        """
        perm = np.arange(len(tensor.get_shape()))
        perm[0] = 1
        perm[1] = 0
        return tf.transpose(tensor, perm=perm)

    @property
    def output_dim(self):
        return self.output_layer.output_dim