import tensorflow as tf
from tensorflow import keras as k
import numpy as np

"""Implementation of the rkn Transition cell, described in 
'Recurrent Kalman Networks:Factorized Inference in High-Dimensional Deep Feature Spaces'
#Todo: add link to paper 
Published at ICML 2019 
Correspondence to: Philipp Becker (philippbecker93@googlemail.com)
"""

# Math Util
def elup1(x):
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return tf.nn.elu(x) + 1


def dadat(A, diag_mat):
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    first_prod = tf.square(A) * diag_ext
    return tf.reduce_sum(first_prod, axis=2)


def dadbt(A, diag_mat, B):
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param B: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    first_prod = A * B * diag_ext
    return tf.reduce_sum(first_prod, axis=2)


# Pack and Unpack functions

def pack_state(mean, covar):
    """ packs system state (either prior or posterior) into single vector
    :param mean: state mean as vector
    :param covar: state covar as list [upper, lower, side]
    :return: state as single vector of size 5 * latent observation dim,
    order of entries: mean, covar_upper, covar_lower, covar_side
    """
    return tf.concat([mean] + covar, -1)


def unpack_state(state):
    """ unpacks system state packed by 'pack_state', can be used to unpack cell output (in non-debug case)
    :param state: packed state, containg mean and covar as single vector
    :return: mean, list of covariances (upper, lower, side)
    """
    lod = int(state.get_shape().as_list()[-1] / 5)
    mean = state[..., :2 * lod]
    covar_upper = state[..., 2 * lod: 3 * lod]
    covar_lower = state[..., 3 * lod: 4 * lod]
    covar_side = state[..., 4*lod:]
    return mean, [covar_upper, covar_lower, covar_side]


def pack_input(obs_mean, obs_covar, obs_valid):
    """ packs cell input. All inputs provided to the cell should be packed using this function
    :param obs_mean: observation mean
    :param obs_covar: observation covariance
    :param obs_valid: flag indication if observation is valid
    :return: packed input
    """
    if not obs_valid.dtype == tf.float32:
        obs_valid = tf.cast(obs_valid, tf.float32)
    return tf.concat([obs_mean, obs_covar, obs_valid], axis=-1)


def unpack_input(input_as_vector):
    """ used to unpack input vectors that where packed with 'pack_input
    :param input_as_vector packed input
    :return: observation mean, observation covar, observation valid flag
    """
    lod = int((input_as_vector.get_shape().as_list()[-1] - 1) / 2)
    obs_mean = input_as_vector[..., :lod]
    obs_covar = input_as_vector[..., lod: -1]
    obs_valid = tf.cast(input_as_vector[..., -1], tf.bool)
    return obs_mean, obs_covar, obs_valid


def pack_debug_output(post_mean, post_covar, prior_mean, prior_covar, kalman_gain):
    """
    packs debug output containing...
    :param post_mean: (vector)
    :param post_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
    :param prior_mean: (vector)
    :param prior_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
    :param kalman_gain: (list of 2 vectors, q_upper, q_lower)
    :return: packed ouptut
    """
    return tf.concat([post_mean] + post_covar, [prior_mean] + prior_covar + kalman_gain, -1)


def unpack_debug_output(output):
    """
    :param output: output produced by the cell in debug mode
    :return: unpacked ouptut, i.e.:
                post_mean: (vector)
                post_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
                prior_mean: (vector)
                prior_covar: (list of 3 vectors, covar_upper, covar_lower, covar_side)
                kalman_gain: (list of 2 vectors, q_upper, q_lower)
    """
    lod = tf.shape(output)[-1] / 12
    post_mean =         output[..., :  2 * lod]
    post_covar_upper =  output[..., 2 * lod:  3 * lod]
    post_covar_lower =  output[..., 3 * lod:  4 * lod]
    post_covar_side =   output[..., 4 * lod:  5 * lod]
    prior_mean =        output[..., 5 * lod:  7 * lod]
    prior_covar_upper = output[..., 7 * lod:  8 * lod]
    prior_covar_lower = output[..., 8 * lod:  9 * lod]
    prior_covar_side =  output[..., 9 * lod: 10 * lod]
    q_upper =           output[..., 10 * lod: 11 * lod]
    q_lower =           output[..., 11 * lod:]
    post_covar = [post_covar_upper, post_covar_lower, post_covar_side]
    prior_covar = [prior_covar_upper, prior_covar_lower, prior_covar_side]
    return post_mean, post_covar, prior_mean, prior_covar, [q_upper. q_lower]


class TransitionNet:
    """Implements a simple dense network, used as coefficient network to get the state dependent coefficentes for the
       transition model """

    def __init__(self, lsd, number_of_basis, hidden_units):
        """
        :param lsd: latent state size (i.e. network input dimension)
        :param number_of_basis: number of basis matrices (i.e. network output dimension)
        :param hidden_units: list of numbers of hidden units
        """
        self._hidden_layers = []
        cur_in_shape = lsd
        for u in hidden_units:
            layer = k.layers.Dense(u, activation=k.activations.relu)
            layer.build([None, cur_in_shape])
            cur_in_shape = u
            self._hidden_layers.append(layer)
        self._out_layer = k.layers.Dense(number_of_basis, activation=k.activations.softmax)
        self._out_layer.build([None, cur_in_shape])

    def __call__(self, latent_state):
        """
        :param latent_state: current latent state
        :return: coefficents for transition basis matrices
        """
        h = latent_state
        for hidden_layer in self._hidden_layers:
            h = hidden_layer(h)
        return self._out_layer(h)

    @property
    def weights(self):
        weigths = self._out_layer.trainable_weights
        for hidden_layer in self._hidden_layers:
            weigths += hidden_layer.trainable_weights
        return weigths


class RKNTransitionCell(k.layers.Layer):
    """Implementation of the actual transition cell. This is implemented as a subclass of the Keras Layer Class, such
     that it can be used with tf.keras.layers.RNN"""

    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 number_of_basis,
                 bandwidth,
                 trans_net_hidden_units=[],
                 initial_trans_covar=0.1,
                 never_invalid=False,
                 debug=False):

        """
        :param latent_state_dim: dimensionality of latent state (n in paper)
        :param latent_obs_dim: dimensionality of latent observation (m in paper)
        :param number_of_basis: number of basis matrices used (k in paper)
        :param bandwidth: bandwidth of transition sub matrices (b in paper)
        :param trans_net_hidden_units: list of number (numbers of hidden units per layer in coefficient network)
        :param initial_trans_covar: value (scalar) used to initialize transition covariance with
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        :param debug: if set the cell output will additionally contain the prior state estimate and kalman gain for
                      debugging/visualization purposes, use 'unpack_debug_output' to unpack.
        """

        super().__init__()

        assert latent_state_dim == 2 * latent_obs_dim, "Currently only 2 * m = n supported"

        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._num_basis = number_of_basis
        self._bandwidth = bandwidth
        self._never_invalid = never_invalid
        self._debug = debug
        self._trans_net_hidden_units = trans_net_hidden_units
        self._initial_trans_covar = initial_trans_covar

    def build(self, input_shape):

        # build state independent basis matrices
        tm_11_init =        np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_12_init =  0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_21_init = -0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_22_init =        np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])

        tm_11_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_11_basis",
                                     initializer=k.initializers.Constant(tm_11_init))
        tm_12_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_12_basis",
                                     initializer=k.initializers.Constant(tm_12_init))
        tm_21_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_21_basis",
                                     initializer=k.initializers.Constant(tm_21_init))
        tm_22_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_22_basis",
                                     initializer=k.initializers.Constant(tm_22_init))

        tm_11, tm_12, tm_21, tm_22 = (tf.linalg.band_part(x, self._bandwidth, self._bandwidth) for x in
                                      [tm_11_full, tm_12_full, tm_21_full, tm_22_full])
        self._basis_matrices = tf.concat([tf.concat([tm_11, tm_12], -1),
                                          tf.concat([tm_21, tm_22], -1)], -2)
        self._basis_matrices = tf.expand_dims(self._basis_matrices, 0)

        # build state dependent coefficent network
        self._coefficient_net = TransitionNet(self._lsd, self._num_basis, self._trans_net_hidden_units)
        self._trainable_weights += self._coefficient_net.weights

        # build learnable state-independent transition covariance - internally work with "log" to ensure positiveness
        elup1_inv = lambda x: (np.log(x) if x < 1.0 else (x - 1.0))
        log_transition_covar = self.add_weight(shape=[1, self._lsd], name="log_transition_covar",
                                               initializer=k.initializers.Constant(elup1_inv(self._initial_trans_covar)))
        trans_covar = elup1(log_transition_covar)
        self._trans_covar_upper = trans_covar[:, :self._lod]
        self._trans_covar_lower = trans_covar[:, self._lod:]

        super().build(input_shape)

    def call(self, inputs, states, **kwargs):
        """Performs one transition step (prediction followed by update in Kalman Filter terms)
        Parameter names match those of superclass - same signature as k.layers.LSTMCell
        :param inputs: Latent Observations (mean and covariance vectors concatenated)
        :param states: Last Latent Posterior State (mean and covariance vectors concatenated)
        :param scope: See super
        :return: cell output: current posterior (if not debug, else current posterior, prior and kalman gain)
                 cell state: current posterior
        """
        # unpack inputs
        obs_mean, obs_covar, obs_valid = unpack_input(inputs)
        state_mean, state_covar = unpack_state(states[0])

        # predict step (next prior from current posterior (i.e. cell state))
        pred_res = self._predict(state_mean, state_covar)
        prior_mean, prior_covar = pred_res

        # update step (current posterior from current prior)
        if self._never_invalid:
            update_res = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        else:
            update_res = self._masked_update(prior_mean, prior_covar, obs_mean, obs_covar, obs_valid)
        post_mean, post_covar = update_res[:2]

        # pack outputs
        post_state = pack_state(post_mean, post_covar)
        if self._debug:
            debug_output = pack_debug_output(post_mean, post_covar, prior_mean, prior_covar, update_res[-1])
            return debug_output, post_state
        else:
            return post_state, post_state

    def _predict(self, post_mean, post_covar):
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        # compute state dependent transition matrix
        coefficients = self._coefficient_net(post_mean)
        scaled_matrices = tf.reshape(coefficients, [-1, self._num_basis, 1, 1]) * self._basis_matrices
        transition_matrix = tf.reduce_sum(scaled_matrices, 1)

        # predict next prior mean
        expanded_state_mean = tf.expand_dims(post_mean, -1)
        new_mean = tf.squeeze(tf.matmul(transition_matrix, expanded_state_mean), -1)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        b11 = transition_matrix[:, :self._lod, :self._lod]
        b12 = transition_matrix[:, :self._lod, self._lod:]
        b21 = transition_matrix[:, self._lod:, :self._lod]
        b22 = transition_matrix[:, self._lod:, self._lod:]

        covar_upper, covar_lower, covar_side = post_covar

        new_covar_upper = dadat(b11, covar_upper) + 2 * dadbt(b11, covar_side, b12) + dadat(b12, covar_lower) \
                          + self._trans_covar_upper
        new_covar_lower = dadat(b21, covar_upper) + 2 * dadbt(b21, covar_side, b22) + dadat(b22, covar_lower) \
                          + self._trans_covar_lower
        new_covar_side = dadbt(b21, covar_upper, b11) + dadbt(b22, covar_side, b11) \
                         + dadbt(b21, covar_side, b12) + dadbt(b22, covar_lower, b12)

        return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def _masked_update(self, prior_mean, prior_covar, obs_mean, obs_covar, obs_valid):
        """ Ensures update only happens if observation is valid
        CAVEAT: You need to ensure that obs_mean and obs_covar do not contain NaNs, even if they are invalid.
        If they do this will cause problems with gradient computation (they will also be NaN) due to how tf.where works
        internally (see: https://github.com/tensorflow/tensorflow/issues/2540)
        :param prior_mean: current prior latent state mean
        :param prior_covar: current prior latent state convariance
        :param obs_mean: current latent observation mean
        :param obs_covar: current latent observation covariance
        :param obs_valid: indicating if observation is valid
        :return: current posterior latent state mean and covariance
        """

        up_res = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        val_mean, val_covar = up_res[:2]
        obs_valid = obs_valid[:, None]
        masked_mean = tf.where(obs_valid, val_mean, prior_mean)

        masked_covar_upper = tf.where(obs_valid, val_covar[0], prior_covar[0])
        masked_covar_lower = tf.where(obs_valid, val_covar[1], prior_covar[1])
        masked_covar_side  = tf.where(obs_valid, val_covar[2], prior_covar[2])

        if self._debug:
            masked_q_upper = tf.where(obs_valid, up_res[-1][0], tf.zeros(tf.shape(obs_mean)))
            masked_q_lower = tf.where(obs_valid, up_res[-1][1], tf.zeros(tf.shape(obs_mean)))
            return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side], \
                   [masked_q_upper, masked_q_lower]
        else:
            return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side]

    def _update(self, prior_mean, prior_covar, obs_mean, obs_covar):
        """Performs update step
        :param prior_mean: current prior latent state mean
        :param prior_covar: current prior latent state covariance
        :param obs_mean: current latent observation mean
        :param obs_covar: current latent covariance mean
        :return: current posterior latent state and covariance
        """
        covar_upper, covar_lower, covar_side = prior_covar

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = covar_upper + obs_covar
        q_upper = covar_upper / denominator
        q_lower = covar_side / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + tf.concat([q_upper * residual, q_lower * residual], -1)

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * covar_upper
        new_covar_lower = covar_lower - q_lower * covar_side
        new_covar_side = covar_factor * covar_side
        if self._debug:
            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side], [q_upper, q_lower]
        else:
            return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]

    def get_initial_state(self, inputs, batch_size, dtype):
        """
        Signature matches the run required by k.layers.RNN
        :param inputs:
        :param batch_size:
        :param dtype:
        :return:
        """
        initial_mean = tf.zeros([batch_size, 2 * self._lod], dtype=dtype)
        initial_covar_diag = 10 * tf.ones([batch_size, 2 * self._lod], dtype=dtype)
        initial_covar_side = tf.zeros([batch_size, 1 * self._lod], dtype=dtype)
        return tf.concat([initial_mean, initial_covar_diag, initial_covar_side], -1)

    @property
    def state_size(self):
        """ required by k.layers.RNN"""
        return 5 * self._lod
