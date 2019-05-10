import tensorflow as tf
from util import MeanCovarPacker as mcp
from util.MulUtil import bmatmul, bmatvec, diag_a_diag_bt, diag_a_diag_at
from transition_cell.TransitionCell import TransitionCell
from network import NDenseHiddenLayers, HiddenLayersParamsKeys, SimpleOutputLayer, FeedForwardNet

class RKNCorrTransitionCell(TransitionCell):
    """Implementation of the Transition Layer described in the paper. This is implemented as a subclass of the
    tensorflow cell and can hence used with tf.nn.dynamic_rnn"""

    def __init__(self,
                 config,
                 transition_matrix,
                 transition_covar,
                 diag_only=False,
                 debug=False):
        """Construct new Transition Cell
        :param config a RKNConfiguration Object
        :param transition_matrix the transition matrix to use
        :param debug: if true additional debugging information is incorporated in the state"""
        super().__init__(config, transition_matrix, transition_covar, diag_only, debug)

        self.observation_matrix = tf.concat([tf.eye(self.latent_observation_dim),
                                             tf.zeros([self.latent_observation_dim, self.latent_observation_dim])], -1)
        self._b11 = self.transition_matrix[:self.latent_observation_dim, :self.latent_observation_dim]
        self._b12 = self.transition_matrix[:self.latent_observation_dim, self.latent_observation_dim:]
        self._b21 = self.transition_matrix[self.latent_observation_dim:, :self.latent_observation_dim]
        self._b22 = self.transition_matrix[self.latent_observation_dim:, self.latent_observation_dim:]

        self._sub_mats = [self._b11, self._b12, self._b21, self._b22]
        self._sq_mats = [tf.transpose(tf.square(x)) for x in self._sub_mats]
        self._prod_mats  = [tf.transpose(self._b11 * self._b12),
                            tf.transpose(self._b21 * self._b22),
                            tf.transpose(self._b21 * self._b11),
                            tf.transpose(self._b22 * self._b11),
                            tf.transpose(self._b21 * self._b12),
                            tf.transpose(self._b22 * self._b12)]



    def _transform_correlated_covar(self, covar_upper, covar_lower, covar_side):
        b11_sq, b12_sq, b21_sq, b22_sq = self._sq_mats
        b11_12, b21_22, b21_11, b22_11, b21_12, b22_12 = self._prod_mats

        #covar_upper, covar_lower, covar_side = [tf.expand_dims(x, 1) for x in [covar_upper, covar_lower, covar_side]]

        new_covar_upper = \
            tf.matmul(covar_upper, b11_sq) + 2 * tf.matmul(covar_side, b11_12) + tf.matmul(covar_lower, b12_sq)
        new_covar_lower = \
            tf.matmul(covar_upper, b21_sq) + 2 * tf.matmul(covar_side, b21_22) + tf.matmul(covar_lower, b22_sq)
        new_covar_side = tf.matmul(covar_upper, b21_11) + tf.matmul(covar_side, b22_11) + \
                         tf.matmul(covar_side, b21_12) + tf.matmul(covar_lower, b22_12)
        return new_covar_upper, new_covar_lower, new_covar_side


    def _predict(self, state_mean, state_covar):
        """ Performs prediction step
        :param state_mean: last posterior mean
        :param state_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        with tf.name_scope('Predict'):
            # The state is transposed since we work with batches (state_mean is of shape [batch_size x latent_state_dim])
            # Hence we do not compute z_new = A*z_old but z_new^T = z_old^T * A^T
            #with tf.control_dependencies([tf.print(state_mean, state_covar)]):
            new_mean = tf.matmul(state_mean, self.transition_matrix, transpose_b=True)
            if isinstance(state_covar, list):
                covar_upper, covar_lower, covar_side = state_covar
            else:
                covar_upper, covar_lower, covar_side = mcp.split_corr_covar(state_covar, self.latent_state_dim)

            trans_covar_raw = self.transition_covar(state_mean) if callable(self.transition_covar) else self.transition_covar
            trans_covar_upper, trans_covar_lower, trans_covar_side = mcp.split_corr_covar(trans_covar_raw, self.latent_state_dim)

            #new_covar_upper_ref, new_covar_lower_ref, new_covar_side_ref = \
            #    self._transform_correlated_covar_ref(covar_upper=covar_upper,
            #                                         covar_lower=covar_lower,
            #                                         covar_side=covar_side,
            #                                         A11=self._b11, A12=self._b12,
            #                                         A21=self._b21, A22=self._b22)
            new_covar_upper, new_covar_lower, new_covar_side = \
                self._transform_correlated_covar(covar_upper=covar_upper,
                                                 covar_lower=covar_lower,
                                                 covar_side=covar_side)

            #with tf.control_dependencies([tf.assert_less(tf.abs(new_covar_upper - new_covar_upper_ref), 1e-5),
            #                              tf.assert_less(tf.abs(new_covar_lower - new_covar_lower_ref), 1e-5),
            #                              tf.assert_less(tf.abs(new_covar_side - new_covar_side_ref), 1e-5)]):

            new_covar_upper += trans_covar_upper
            new_covar_lower += trans_covar_lower
            new_covar_side += trans_covar_side

            if self.debug:
                return new_mean, [new_covar_upper, new_covar_lower, new_covar_side], trans_covar_raw
            else:
                return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def _masked_update(self, state_mean, state_covar, observation_mean, observation_covar, observation_valid):
        """ Ensures update only happens if observation is valid
        :param state_mean: current prior latent state mean
        :param state_covar: current prior latent state convariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent observation covariance
        :param observation_valid: indicating if observation is valid
        :return: current posterior latent state mean and covariance
        """
        # TODO Reimplement without cond, not supported on gpu
        assert not self.debug, "Currently no masked update and debug"
        a = self._update(state_mean, state_covar, observation_mean, observation_covar)
        return tf.where(observation_valid, a[0], state_mean), \
               [tf.where(observation_valid, a[1][0], state_covar[0]),
                tf.where(observation_valid, a[1][1], state_covar[1]),
                tf.where(observation_valid, a[1][2], state_covar[2])]


    def _update(self, state_mean, state_covar, observation_mean, observation_covar):
        with tf.name_scope("Update"):
            covar_upper, covar_lower, covar_side = state_covar
            with tf.name_scope("KalmanGain"):
                denominator = covar_upper + observation_covar
                q_upper = covar_upper / denominator
                q_lower = covar_side / denominator
            residual = observation_mean - state_mean[:, :self.latent_observation_dim]
            new_mean = state_mean + tf.concat([q_upper * residual, q_lower * residual], -1)

            covar_factor = 1 - q_upper
            new_covar_upper = covar_factor * covar_upper
          #  with tf.control_dependencies([tf.assert_greater(new_covar_upper, tf.constant(0, dtype=tf.float32), message="update upper")]):
          #      new_covar_upper = tf.identity(new_covar_upper)
            new_covar_lower = covar_lower - q_lower * covar_side
          #  with tf.control_dependencies([tf.assert_greater(new_covar_lower, tf.constant(0, dtype=tf.float32), message="update lower")]):
          #      new_covar_lower = tf.identity(new_covar_lower)
            new_covar_side  = covar_factor * covar_side
            if self.debug:
                return new_mean, [new_covar_upper, new_covar_lower, new_covar_side], tf.concat([q_upper, q_lower], -1)
            else:
                return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]



    @property
    def _lower_dim(self):
        """The dimensionality of the memory part of the latent state (lower part if written as column vector)
         Currently this should always be equal to self.latent_observation_dim"""
        return self.latent_state_dim - self.latent_observation_dim

    def _normalize_if_desired(self, means, covars, flag):
        """
        Normalizes a batch of vectors (across last axis) iff self.normalize
        :param mean: batch of vector to normalize
        :return: the normalized batch
        """
        if self.c.normalize_latent and flag:
            self.c.mprint("true")
            if self.c.adapt_variance_to_normalization:
                normalized, d_normalized = self._norm_and_derivative(means)
                if not isinstance(covars, list):
                    transformed_covar = diag_a_diag_at(d_normalized, covars)
                else:
                    transformed_covar = \
                        self._transform_correlated_covar(covar_upper=covars[0], covar_lower=covars[1], covar_side=covars[2],
                                                         b11=d_normalized[:, :self.latent_observation_dim, :self.latent_observation_dim],
                                                         A12=d_normalized[: ,:self.latent_observation_dim, self.latent_observation_dim:],
                                                         A21=d_normalized[:, self.latent_observation_dim:, :self.latent_observation_dim],
                                                         A22=d_normalized[:, self.latent_observation_dim:, self.latent_observation_dim:])

                return normalized, transformed_covar
            else:
                if self.c.use_sigmoid_in_normalization:
                    means = tf.nn.sigmoid(means)
                return means / tf.norm(means, ord='euclidean', axis=-1, keepdims=True), covars
        else:
            self.c.mprint("false")
            return means, covars

    def _norm_and_derivative(self, value):
        batch_size = tf.shape(value)[0]
        s = tf.norm(value, ord='euclidean', axis=-1, keepdims=True)

        normalized = value / s
        normalized_exp = tf.expand_dims(normalized, -1)
        temp = tf.matmul(normalized_exp, normalized_exp, transpose_b=True)
        d_normalized = (1 / tf.expand_dims(s, -1)) * (tf.eye(tf.shape(value)[-1], batch_shape=[batch_size]) - temp)

        return normalized, d_normalized

    def _transform_correlated_covar_ref(self, covar_upper, covar_lower, covar_side, A11, A12, A21, A22):
        new_covar_upper = diag_a_diag_at(A11, covar_upper) + \
                          2 * diag_a_diag_bt(A11, covar_side, A12) + \
                          diag_a_diag_at(A12, covar_lower)
        new_covar_lower = diag_a_diag_at(A21, covar_upper) + \
                          2 * diag_a_diag_bt(A21, covar_side, A22) + \
                          diag_a_diag_at(A22, covar_lower)
        new_covar_side = diag_a_diag_bt(A21, covar_upper, A11) + \
                         diag_a_diag_bt(A22, covar_side, A11)  + \
                         diag_a_diag_bt(A21, covar_side, A12)  + \
                         diag_a_diag_bt(A22, covar_lower, A12)
        return new_covar_upper, new_covar_lower, new_covar_side

    """Superclass methods"""

    def pack_state(self, mean, covar):
        return mcp.pack_corr_covar(mean, covar)

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
        if self.debug:
            return 2 * (self.latent_state_dim + 3 * self.latent_observation_dim) +  5 * self.latent_observation_dim
        else:
            return 2 * (self.latent_state_dim + 3 * self.latent_observation_dim)

    @property
    def state_size(self):
        """see super"""
        return self.latent_state_dim + 3 * self.latent_observation_dim

    def zero_state(self, batch_size, dtype):
        """see super"""
        return tf.zeros([batch_size, self.state_size], dtype=dtype)

    """testing"""

    def _update_ineff(self, state_mean, state_covar, observation_mean, observation_covar):
        """Performs update step
        :param state_mean: current prior latent state mean
        :param state_covar: current prior latent state covariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent covariance mean
        :return: current posterior latent state and covariance
        """
        with tf.name_scope("Update"):
            batch_size = tf.shape(state_mean)[0]
            state_covar = self._expand_covar(state_covar)
            with tf.name_scope("KalmanGain"):
                kg_nominator = bmatmul(state_covar, self.observation_matrix, transpose_b=True)
                kg_denominator = bmatmul(self.observation_matrix, kg_nominator) + tf.matrix_diag(observation_covar)

                kalman_gain = tf.matrix_transpose(tf.matrix_solve(tf.matrix_transpose(kg_denominator),
                                                                  tf.matrix_transpose(kg_nominator)))

                #kalman_gain = tf.Print(kalman_gain, [kalman_gain], "kg")

            residual = observation_mean - tf.matmul(state_mean, self.observation_matrix, transpose_b=True)
            new_mean = state_mean + bmatvec(kalman_gain, residual)

            covar_factor = tf.eye(self.latent_state_dim, batch_shape=[batch_size]) - bmatmul(kalman_gain, self.observation_matrix)
            new_covar = tf.matmul(covar_factor, state_covar)
            return new_mean, self._compress_covar(new_covar)

    def _predict_ineff(self, state_mean, state_covar):

        with tf.name_scope('Predict'):
            # The state is transposed since we work with batches (state_mean is of shape [batch_size x latent_state_dim])
            # Hence we do not compute z_new = A*z_old but z_new^T = z_old^T * A^T
            new_mean = tf.matmul(state_mean, self.transition_matrix, transpose_b=True)
            m1 = bmatmul(self.transition_matrix, self._expand_covar(state_covar))
            m2 = bmatmul(m1, self.transition_matrix, transpose_b=True)
            new_covar = m2 + self._expand_covar(self.transition_covar)
            new_covar_upper, new_covar_lower, new_covar_side = self._compress_covar(new_covar)

            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def _expand_covar(self, covar):
        if not isinstance(covar, list):
            covar_list = mcp.split_corr_covar(covar, self.latent_state_dim)
        else:
            covar_list = covar
        covar_upper = tf.matrix_diag(covar_list[0])
        covar_lower = tf.matrix_diag(covar_list[1])
        covar_side = tf.matrix_diag(covar_list[2])
        upper = tf.concat([covar_upper, covar_side], -1)
        lower = tf.concat([covar_side, covar_lower], -1)
        full = tf.concat([upper, lower], -2)
        #    full = tf.Print(full, [covar_upper, covar_lower, covar_side], "expand uls")
        return full

    def _compress_covar(self, full_covar):
        covar_upper = tf.matrix_diag_part(full_covar[..., :self.latent_observation_dim, :self.latent_observation_dim])
        covar_lower = tf.matrix_diag_part(full_covar[..., self.latent_observation_dim:, self.latent_observation_dim:])
       # covar_lower = tf.Print(covar_lower, [covar_lower], summarize=8)
  #      with tf.control_dependencies([tf.assert_greater(covar_upper, tf.constant(0, dtype=tf.float32))]):
        covar_side = tf.matrix_diag_part(full_covar[..., self.latent_observation_dim:, :self.latent_observation_dim])
        #     covar_side = tf.Print(covar_side, [covar_upper, covar_lower, covar_side], "compress uls")
        return [covar_upper, covar_lower, covar_side]
