import tensorflow as tf
from transition_cell.TransitionCell import TransitionCell
from util import MeanCovarPacker as mcp
from util import MulUtil as mul

class RKNSimpleTransitionCell(TransitionCell):
    """Implementation of the Transition Layer described in the paper. This is implemented as a subclass of the
    tensorflow cell and can hence used with tf.nn.dynamic_rnn"""

    def _predict(self, state_mean, state_covar):
        """ Performs prediction step
        :param state_mean: last posterior mean
        :param state_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        with tf.name_scope('Predict'):
            # The state is transposed since we work with batches (state_mean is of shape [batch_size x latent_state_dim])
            # Hence we do not compute z_new = A*z_old but z_new^T = z_old^T * A^T
            new_mean = tf.matmul(state_mean, self.transition_matrix, transpose_b=True)
            trans_covar = self.transition_covar(state_mean) if callable(self.transition_covar) else self.transition_covar
            new_covar = mul.diag_a_diag_at(self.transition_matrix, state_covar) + trans_covar
            return new_mean, new_covar

    def _update(self, state_mean, state_covar, observation_mean, observation_covar):
        """Performs update step
        :param state_mean: current prior latent state mean
        :param state_covar: current prior latent state covariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent covariance mean
        :return: current posterior latent state and covariance
        """
        with tf.name_scope("Update"):
            batch_size = tf.shape(state_mean)[0]
            with tf.name_scope("KalmanGain"):
                # sigma_trans * H^T, with H = (I 0)
                kg_nominator = state_covar[:, :self.latent_observation_dim]
                # H * sigma_trans * H^T + sigma_obs, with H = (I 0)
                kg_denominator = state_covar[:, :self.latent_observation_dim] + observation_covar
                kalman_gain_upper = kg_nominator / kg_denominator

            # r = w_t - H * z_t, with H = (I 0)
            residual = observation_mean - state_mean[:, :self.latent_observation_dim]
            update_upper = kalman_gain_upper * residual
            # This can be simplified if TensorFlow one day supports item assignments of Tensors
            update_lower = tf.zeros((batch_size, self._lower_dim))
            update = tf.concat([update_upper, update_lower], -1)
            new_mean = state_mean + update

            covar_factor_upper = 1.0 - kalman_gain_upper

            new_covar_upper = covar_factor_upper * state_covar[:, :self.latent_observation_dim]
            new_covar = tf.concat([new_covar_upper, state_covar[:, self.latent_observation_dim:]], -1)

            if self.debug:
                return new_mean, new_covar, kalman_gain_upper
            else:
                return new_mean, new_covar



    @property
    def _lower_dim(self):
        """The dimensionality of the memory part of the latent state (lower part if written as column vector)
         Currently this should always be equal to self.latent_observation_dim"""
        return self.latent_state_dim - self.latent_observation_dim

    def _normalize_if_desired(self, batch_of_vectors):
        """
        Normalizes a batch of vectors (across last axis) iff self.normalize
        :param batch_of_vectors: batch of vector to normalize
        :return: the normalized batch
        """
        if self.c.normalize_latent:
            return batch_of_vectors / tf.norm(batch_of_vectors, ord='euclidean', axis=-1, keepdims=True)
        else:
            return batch_of_vectors

    """Superclass methods"""
    def pack_state(self, mean, covar):
        return mcp.pack(mean, covar)


    def unpack_state(self, state_as_vector):
        return mcp.unpack(state_as_vector, self.latent_state_dim)

    def pack_input(self, observation_mean, observation_covar, observation_valid):
        obs = mcp.pack(observation_mean, observation_covar)
        if not observation_valid.dtype == tf.float32:
            observation_valid = tf.cast(observation_valid, tf.float32)
        return tf.concat([obs, observation_valid], -1)

    def unpack_input(self, input_as_vector):
        obs_mean, obs_covar = mcp.unpack(input_as_vector[:, :-1], self.latent_observation_dim)
        observation_valid = tf.cast(input_as_vector[:, -1], tf.bool)
        return obs_mean, obs_covar, observation_valid

    @property
    def output_size(self):
        """see super"""
        return 2 * self.latent_state_dim if not self.debug else (4 * self.latent_state_dim) + self.latent_observation_dim

    @property
    def state_size(self):
        """see super"""
        return 2 * self.latent_state_dim

    def zero_state(self, batch_size, dtype):
        """see super"""
        return tf.zeros([batch_size, self.state_size], dtype=dtype)

