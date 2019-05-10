import tensorflow as tf
from util import MeanCovarPacker as mcp
from util.MulUtil import bmatmul, bmatvec
from transition_cell.TransitionCell import TransitionCell

class RKNFullTransitionCell(TransitionCell):
    """Implementation of the Transition Layer described in the paper. This is implemented as a subclass of the
    tensorflow cell and can hence used with tf.nn.dynamic_rnn"""

    def __init__(self,
                 config,
                 transition_matrix,
                 transition_covar,
                 debug=False):
        """Construct new Transition Cell
        :param config a RKNConfiguration Object
        :param transition_matrix the transition matrix to use
        :param debug: if true additional debugging information is incorporated in the state"""
        super().__init__(config, transition_matrix, transition_covar, False, debug)
        if self.debug:
            tf.logging.warn("Debug currently not correctly implemented")

    def _transition(self, state_mean, state_covar, observation_mean, observation_covar, observation_valid):
        """Performs transition step if last latent state is given. Assumes input state to be normalized!
        :param state_mean: last latent state mean
        :param state_covar: last latent state covariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent observation covariance
        :param observation_valid: indicating if observation is valid
        :return: Next state
        """
        with tf.name_scope('Transition'):
            prior_state_mean, prior_state_covariance = self._predict(state_mean,
                                                                     state_covar)

            prior_state_mean = self._normalize_if_desired(prior_state_mean)
            observation_mean = self._normalize_if_desired(observation_mean)

            if self.c.never_invalid:
                update_res = self._update(prior_state_mean, prior_state_covariance,
                                          observation_mean, observation_covar)
            else:
                update_res = self._masked_update(prior_state_mean, prior_state_covariance,
                                                 observation_mean, observation_covar, observation_valid)
            post_state_mean, post_state_covariance = update_res

            post_state_mean = self._normalize_if_desired(post_state_mean)
            post_state = self.pack_state(post_state_mean, post_state_covariance)
            return post_state, post_state


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
            tm_exp = tf.tile(tf.expand_dims(self.transition_matrix, 0), [tf.shape(state_covar)[0], 1, 1])
            m1 = tf.matmul(tm_exp, state_covar)
            m2 = tf.matmul(m1, tf.transpose(tm_exp, [0, 2, 1]))

            #m1 = bmatmul(self.transition_matrix, state_covar)
            #m2 = bmatmul(m1, self.transition_matrix, transpose_b=True)


            trans_covar = self.transition_covar(state_mean) if callable(self.transition_covar) else self.transition_covar
            new_covar = m2 + trans_covar
           # with tf.control_dependencies([tf.assert_greater(tf.matrix_diag_part(new_covar), tf.constant(0, dtype=tf.float32), message="Predict")]):
           #     new_covar = tf.identity(new_covar)
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
                kg_nominator = state_covar[:, :, :self.latent_observation_dim]
                #       observation_covar = tf.Print(observation_covar, [observation_covar], "obs_covar")
                kg_denominator = state_covar[:, :self.latent_observation_dim, :self.latent_observation_dim] + observation_covar

                kalman_gain = tf.matrix_transpose(tf.matrix_solve(tf.matrix_transpose(kg_denominator),
                                                                  tf.matrix_transpose(kg_nominator)))

            residual = observation_mean - state_mean[:, :self.latent_observation_dim]
            new_mean = state_mean + bmatvec(kalman_gain, residual)

            covar_factor = tf.eye(self.latent_state_dim, batch_shape=[batch_size]) \
                           - tf.concat([kalman_gain, tf.zeros([batch_size, self.latent_state_dim, self.latent_observation_dim])], -1)

          #  covar_factor = tf.Print(covar_factor, [covar_factor], message="bla")
            new_covar = tf.matmul(covar_factor, state_covar)
           # new_covar = tf.matmul(covar_factor, tf.matmul(state_covar, covar_factor, transpose_b=True)) + \
          #              tf.matmul(kalman_gain, tf.matmul(observation_covar, kalman_gain, transpose_b=True))
         #   with tf.control_dependencies([tf.assert_greater(tf.matrix_diag_part(new_covar), tf.constant(0, dtype=tf.float32), message="Update")]):
         #       new_covar = tf.identity(new_covar)
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
        return mcp.pack_full_covar(mean, covar)


    def unpack_state(self, state_as_vector):
        return mcp.unpack_full_covar(state_as_vector, self.latent_state_dim)


    def pack_input(self, observation_mean, observation_covar, observation_valid):
        obs = mcp.pack_full_covar(observation_mean, observation_covar)
        if not observation_valid.dtype == tf.float32:
            observation_valid = tf.cast(observation_valid, tf.float32)
        return tf.concat([obs, observation_valid], -1)

    def unpack_input(self, input_as_vector):
        obs_mean, obs_covar = mcp.unpack_full_covar(input_as_vector[:, :-1], self.latent_observation_dim)
        obs_valid = tf.cast(input_as_vector[:, -1], tf.bool)
        return obs_mean, obs_covar, obs_valid

    @property
    def output_size(self):
        """see super"""
        return self.latent_state_dim + self.latent_state_dim ** 2

    @property
    def state_size(self):
        """see super"""
        return self.latent_state_dim + self.latent_state_dim ** 2

    def zero_state(self, batch_size, dtype):
        """see super"""
        return tf.zeros([batch_size, self.state_size], dtype=dtype)