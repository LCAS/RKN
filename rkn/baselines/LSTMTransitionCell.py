import tensorflow as tf
import util.MeanCovarPacker as mcp

class LSTMTransitionCell(tf.nn.rnn_cell.LSTMCell):


    def pack_input(self, observation_mean, observation_covar, observation_valid):
        obs = mcp.pack(observation_mean, observation_covar)
        if not observation_valid.dtype == tf.float32:
            observation_valid = tf.cast(observation_valid, tf.float32)
        return tf.concat([obs, observation_valid], -1)

    def set_state_dim(self, state_dim):
        self.state_dim = state_dim

    def unpack_state(self, state_as_vector):
        mean, log_covar = mcp.unpack(state_as_vector, self.state_dim)
        return mean, tf.exp(log_covar)
