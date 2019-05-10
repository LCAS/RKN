import tensorflow as tf
from transition_cell.RKNCorrTransitionCell import RKNCorrTransitionCell
import util.MeanCovarPacker as mcp
from util.MulUtil import diag_a_diag_at_batched, diag_a_diag_bt_batched

class LLRKNTransitionCell(RKNCorrTransitionCell):
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

            b11 = transition_matrix[:, :self.latent_observation_dim, :self.latent_observation_dim]
            b12 = transition_matrix[:, :self.latent_observation_dim, self.latent_observation_dim:]
            b21 = transition_matrix[:, self.latent_observation_dim:, :self.latent_observation_dim]
            b22 = transition_matrix[:, self.latent_observation_dim:, self.latent_observation_dim:]
            #trans_covar = self.transition_covar(state_mean) if callable(self.transition_covar) else self.transition_covar
            if isinstance(state_covar, list):
                covar_upper, covar_lower, covar_side = state_covar
            else:
                covar_upper, covar_lower, covar_side = mcp.split_corr_covar(state_covar, self.latent_state_dim)
            new_covar_upper, new_covar_lower, new_covar_side = \
                self._transform_correlated_covar(covar_upper=covar_upper,
                                                 covar_lower=covar_lower,
                                                 covar_side=covar_side,
                                                 b11=b11, A12=b12,
                                                 A21=b21, A22=b22)
            trans_covar_upper, trans_covar_lower, trans_covar_side = mcp.split_corr_covar(trans_covar, self.latent_state_dim)
            new_covar_upper += trans_covar_upper
            new_covar_lower += trans_covar_lower
            new_covar_side += trans_covar_side

    #            new_covar = tf.matmul(transition_matrix, tf.matmul(ex_covar, transition_matrix, transpose_b=True)) + trans_covar
            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def _transform_correlated_covar(self, covar_upper, covar_lower, covar_side, b11, A12, A21, A22):
        new_covar_upper = diag_a_diag_at_batched(b11, covar_upper) + \
                          2 * diag_a_diag_bt_batched(b11, covar_side, A12) + \
                          diag_a_diag_at_batched(A12, covar_lower)
        new_covar_lower = diag_a_diag_at_batched(A21, covar_upper) + \
                          2 * diag_a_diag_bt_batched(A21, covar_side, A22) + \
                          diag_a_diag_at_batched(A22, covar_lower)
        new_covar_side = diag_a_diag_bt_batched(A21, covar_upper, b11) + \
                         diag_a_diag_bt_batched(A22, covar_side, b11) + \
                         diag_a_diag_bt_batched(A21, covar_side, A12) + \
                         diag_a_diag_bt_batched(A22, covar_lower, A12)
        return new_covar_upper, new_covar_lower, new_covar_side

    def _get_ll_transition_matrix(self, state_mean):
        coefficients = self.coefficient_fn(state_mean)
      #  coefficients = tf.Print(coefficients, [coefficients], message="coeff", summarize=32)
        scaled_matrices = coefficients * self.basis_matrices
        transition_matrices = tf.reduce_sum(scaled_matrices, 1)
        return transition_matrices


