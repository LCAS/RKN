import tensorflow as tf
from util.MulUtil import diag_a_diag_at


class LinearDecoder:

    def __init__(self, input_dim, output_dim):
        print("Building Linear decoder")
        self._w = tf.get_variable(name="w",
                                  shape=[input_dim, output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer())

        self._b = tf.get_variable(name="b",
                                  shape=[output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())

    def __call__(self, inputs, training, sequence_data=False):
        if sequence_data:
            return tf.concat(tf.map_fn(lambda x: tf.nn.xw_plus_b(x, self._w, self._b), inputs), 0)
        else:
            return tf.nn.xw_plus_b(inputs, self._w, self._b)

    def decode_diagonal_variance(self, variance_diagonal, training, sequence_data=False):
        w_t = tf.matrix_transpose(self._w)
        if sequence_data:
            return tf.concat(tf.map_fn(lambda x: diag_a_diag_at(w_t, x), variance_diagonal), 0)
        else:
            return diag_a_diag_at(w_t, variance_diagonal)

class FixedLinearDecoder(LinearDecoder):
    def __init__(self, w, b=None):
        self._w = w
        self._b = b if b is not None else tf.zeros([tf.shape(w)[1]], dtype=tf.float32)
