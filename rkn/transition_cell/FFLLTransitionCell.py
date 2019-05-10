from transition_cell.RKNCorrTransitionCell import RKNCorrTransitionCell
import tensorflow as tf
from util.MeanCovarPacker import split_corr_covar


class FFLLTransitionCell(RKNCorrTransitionCell):

    def __init__(self, config, networks, transition_covar=None, debug=False, state_dependent_variance=True):
        if not state_dependent_variance:
            assert transition_covar is not None, "transition covar needs to be given if not state dependent"
        super().__init__(config=config,
                         # dummy, not used!
                         transition_matrix=tf.eye(config.latent_state_dim), transition_covar=transition_covar,
                         diag_only=True, debug=debug)

        self._state_dependent_variance = state_dependent_variance
        self.c = config
        self._networks = networks

    def _predict(self, state_mean, state_covar):

        tm_outputs = []
        if self._state_dependent_variance:
            tc_outputs = []
        for i, network in enumerate(self._networks):
            current_in = tf.stack([state_mean[:,i],
                                   state_mean[:, self.latent_observation_dim + i]], axis=-1)
                                   #state_covar[:, i],
                                   #state_covar[:, 1 *self.latent_observation_dim + i],
                                   #state_covar[:, 2 *self.latent_observation_dim + i]], axis=-1)
            out = network(current_in)
            if self._state_dependent_variance:
                tm_outputs.append(out[0])
                tc_outputs.append(out[1])
            else:
                tm_outputs.append(out)
        tm_11 = tf.stack([tm_outputs[i][:, 0] for i in range(self.latent_observation_dim)], -1)
        tm_12 = tf.stack([tm_outputs[i][:, 1] for i in range(self.latent_observation_dim)], -1)
        tm_21 = tf.stack([tm_outputs[i][:, 2] for i in range(self.latent_observation_dim)], -1)
        tm_22 = tf.stack([tm_outputs[i][:, 3] for i in range(self.latent_observation_dim)], -1)

        if self._state_dependent_variance:
            tc_u = tf.stack([tc_outputs[i][:, 0] for i in range(self.latent_observation_dim)], -1)
            tc_l = tf.stack([tc_outputs[i][:, 1] for i in range(self.latent_observation_dim)], -1)
        else:
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

