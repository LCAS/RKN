import tensorflow as tf
from transition_cell.RKNFullTransitionCell import RKNFullTransitionCell
import util.MeanCovarPacker as mcp
from util.MulUtil import diag_a_diag_at_batched, diag_a_diag_bt_batched

class LLRKNTransitionCellFull(RKNFullTransitionCell):
    """Implementation of the Transition Layer described in the paper. This is implemented as a subclass of the
    tensorflow cell and can hence used with tf.nn.dynamic_rnn"""


    """The four modes of initializing the cell, 1 & 2 are described in the paper:
    1) Copy the observation from the encoder, rest random"""
    INIT_MODE_COPY_OBSERVATION = 'init_copy'
    """2) All random"""
    INIT_MODE_RANDOM = 'init_random'
    """3) Learned - with the current implementation the mean is fixed and a single variance value for all units is learned"""
    INIT_MODE_LEARNED = 'init_learn'
    """4) Constant"""
    INIT_MODE_CONSTANT = 'init_constant'
    def __init__(self,
                 config,
                 basis_matrices,
                 coefficenet_fn,
                 transition_covar,
                 debug=False):
        #         transition_noise_var):
        """Construct new Transition Cell
        :param config a RKNConfiguration Object
        :param transition_matrix the transition matrix to use
        :param debug: if true additional debugging information is incorporated in the state"""
        self.c = config
        self.latent_state_dim = config.latent_state_dim
        self.latent_observation_dim = config.latent_observation_dim

        self.basis_matrices = tf.expand_dims(basis_matrices, 0)
        self.coefficient_fn = lambda x: tf.expand_dims(tf.expand_dims(coefficenet_fn(x), -1), -1)

        self.transition_covar = transition_covar

        self.debug = debug

        assert self.latent_state_dim == 2 * self.latent_observation_dim, "Currently only 2 * m = n supported"

    def _predict(self, state_mean, state_covar):
        """ Performs prediction step
        :param state_mean: last posterior mean
        :param state_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        with tf.name_scope('Predict'):
            transition_matrix = self._get_ll_transition_matrix(state_mean)
            expanded_state_mean = tf.expand_dims(state_mean, -1)
            expanded_new_mean = tf.matmul(transition_matrix, expanded_state_mean)
            new_mean = tf.squeeze(expanded_new_mean, -1)
            if callable(self.transition_covar):
                trans_covar = self.transition_covar(state_mean)
            else:
                trans_covar = self.transition_covar

               # The state is transposed since we work with batches (state_mean is of shape [batch_size x latent_state_dim])
                # Hence we do not compute z_new = A*z_old but z_new^T = z_old^T * A^T
            m1 = tf.matmul(transition_matrix, state_covar)
            m2 = tf.matmul(m1, tf.transpose(transition_matrix, [0, 2, 1]))
            new_covar = m2 + trans_covar
                # with tf.control_dependencies([tf.assert_greater(tf.matrix_diag_part(new_covar), tf.constant(0, dtype=tf.float32), message="Predict")]):
                #     new_covar = tf.identity(new_covar)

            #            new_covar = tf.matmul(transition_matrix, tf.matmul(ex_covar, transition_matrix, transpose_b=True)) + trans_covar
            return new_mean, new_covar

    def _transform_correlated_covar(self, covar_upper, covar_lower, covar_side, A11, A12, A21, A22):
        new_covar_upper = diag_a_diag_at_batched(A11, covar_upper) + \
                          2 * diag_a_diag_bt_batched(A11, covar_side, A12) + \
                          diag_a_diag_at_batched(A12, covar_lower)
        new_covar_lower = diag_a_diag_at_batched(A21, covar_upper) + \
                          2 * diag_a_diag_bt_batched(A21, covar_side, A22) + \
                          diag_a_diag_at_batched(A22, covar_lower)
        new_covar_side = diag_a_diag_bt_batched(A21, covar_upper, A11) + \
                         diag_a_diag_bt_batched(A22, covar_side, A11)  + \
                         diag_a_diag_bt_batched(A21, covar_side, A12)  + \
                         diag_a_diag_bt_batched(A22, covar_lower, A12)
        return new_covar_upper, new_covar_lower, new_covar_side

    def _get_ll_transition_matrix(self, state_mean):
        coefficients = self.coefficient_fn(state_mean)
      #  coefficients = tf.Print(coefficients, [coefficients], message="coeff", summarize=32)
        scaled_matrices = coefficients * self.basis_matrices
        transition_matrices = tf.reduce_sum(scaled_matrices, 1)
        return transition_matrices

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

