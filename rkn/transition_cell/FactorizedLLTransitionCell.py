from transition_cell.RKNCorrTransitionCell import RKNCorrTransitionCell
import tensorflow as tf
from util.MeanCovarPacker import split_corr_covar


class FactorizedLLTransitionCell(RKNCorrTransitionCell):

    def __init__(self, config, network, transition_covar=None, debug=False, state_dependent_variance=True):
        if not state_dependent_variance:
            assert transition_covar is not None, "transition covar needs to be given if not state dependent"
        super().__init__(config=config,
                         # dummy, not used!
                         transition_matrix=tf.eye(config.latent_state_dim), transition_covar=transition_covar,
                         diag_only=True, debug=debug)

        self._state_dependent_variance = state_dependent_variance
        self.c = config
        self._network = network

    def _predict(self, state_mean, state_covar):

#        state_mean = tf.Print(state_mean, [state_mean], "mean", summarize=20)
#        state_covar = tf.Print(state_covar, [state_covar], "covar", summarize=20)
        if self._state_dependent_variance:
            tm_11, tm_12, tm_21, tm_22, tc_u, tc_l = self._network(state_mean)
     #       tm_11 = tf.Print(tm_11, [tm_11], "tm_11", summarize=20)
     #       tm_12 = tf.Print(tm_12, [tm_12], "tm_12", summarize=20)
    #        tm_21 = tf.Print(tm_21, [tm_21], "tm_21", summarize=20)
    #        tm_22 = tf.Print(tm_22, [tm_22], "tm_22", summarize=20)
#            tc_u = tf.Print(tc_u, [tc_u], "tc_u", summarize=20)
#            tc_l = tf.Print(tc_l, [tc_l], "tc_l", summarize=20)

        else:
            tm_11, tm_12, tm_21, tm_22 = self._network(state_mean)
  #          tm_11 = tf.Print(tm_11, [tm_11], "tm_11", summarize=20)
  #          tm_12 = tf.Print(tm_12, [tm_12], "tm_12", summarize=20)
    #        tm_21 = tf.Print(tm_21, [tm_21], "tm_21", summarize=20)
   #         tm_22 = tf.Print(tm_22, [tm_22], "tm_22", summarize=20)
            tc_u, tc_l, tc_s = split_corr_covar(self.transition_covar, self.latent_state_dim)

        s_u, s_l = state_mean[:, :self.latent_observation_dim], state_mean[:, self.latent_observation_dim:]
        c_u, c_l, c_s = split_corr_covar(state_covar, self.latent_state_dim)

        ns_u = tm_11 * s_u + tm_12 * s_l
        ns_l = tm_21 * s_u + tm_22 * s_l
        new_mean = tf.concat([ns_u, ns_l], -1)

        nc_u = c_u * (tm_11 ** 2) + 2 * tm_11 * c_s * tm_12 + c_l * (tm_12 ** 2) + tc_u
        nc_l = c_u * (tm_21 ** 2) + 2 * tm_21 * c_s * tm_22 + c_l * (tm_22 ** 2) + tc_l
        nc_s = tm_21 * c_u * tm_11 + tm_22 * c_s * tm_11 + tm_21 * c_s * tm_12 + tm_22 * c_l * tm_12

        if self.debug:
            return new_mean, \
                   [nc_u, nc_l, nc_s],\
                   tf.concat([tc_u, tc_l, tf.zeros([tf.shape(tc_u)[0], self.latent_observation_dim])], -1)
        else:
            return new_mean, [nc_u, nc_l, nc_s]
"""    def _split_raw(self, raw):
        tm_11 = (raw[:, 0 * self.latent_observation_dim: 1 * self.latent_observation_dim] / 1000) + 0.95 *tf.ones([1, self.latent_observation_dim])
        tm_12 = raw[:, 1 * self.latent_observation_dim: 2 * self.latent_observation_dim] / 1000
        tm_21 = raw[:, 2 * self.latent_observation_dim: 3 * self.latent_observation_dim] / 1000
        tm_22 = (raw[:, 3 * self.latent_observation_dim: 4 * self.latent_observation_dim] / 1000) + 0.95 * tf.ones([1, self.latent_observation_dim])
        tc_u  = self.c.variance_activation_fn(raw[:, 4 * self.latent_observation_dim: 5 * self.latent_observation_dim] / 1000)
        tc_l  = self.c.variance_activation_fn(raw[:, 5 * self.latent_observation_dim: 6 * self.latent_observation_dim] / 1000)
        return tm_11, tm_12, tm_21, tm_22, tc_u, tc_l"""

