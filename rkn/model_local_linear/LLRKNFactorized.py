import tensorflow as tf
from model.RKN import RKN
import numpy as np
from network.OutputLayer import OutputLayer, SimpleOutputLayer
from network.HiddenLayers import  NDenseHiddenLayers
from network.FeedForwardNet import FeedForwardNet
from transition_cell.FactorizedLLTransitionCell import FactorizedLLTransitionCell

class LLRKNFactorized(RKN):

    def _build_transition_cell(self):
        self.c.mprint("Using Fully Factorized LL-RKN Transition Cell")
        with tf.name_scope("TransitionModel"):
            if self.c.transition_network_hidden_dict is not None:
                transition_hidden = NDenseHiddenLayers(params=self.c.transition_network_hidden_dict,
                                                       name_prefix='TransitionHidden')
            else:
                transition_hidden = None
            transition_out = FactorizedLLRKNOutputLayer(latent_observation_dim=self.c.latent_observation_dim,
                                                        name_prefix='TransitionOut',
                                                        with_variance=self.c.learn_state_dependent_transition_covar,
                                                        var_activation=self.c.variance_activation_fn)
            transition_net = FeedForwardNet(transition_out, transition_hidden)

            self.transition_cell = FactorizedLLTransitionCell(config=self.c,
                                                              network=transition_net,
                                                              transition_covar=self._transition_covar,
                                                              state_dependent_variance=self.c.learn_state_dependent_transition_covar,
                                                              debug=self.debug_recurrent)

class LLRKNFactorizedBasis(RKN):

    def _build_transition_cell(self):

        with tf.name_scope("TransitionModel"):
            diag_init = np.expand_dims(np.ones([self.c.num_basis, self.latent_observation_dim]), 0)
            tm12_init = 0.2 * np.random.rand(1, self.c.num_basis, self.latent_observation_dim)
            tm21_init = -0.2 * np.random.rand(1, self.c.num_basis, self.latent_observation_dim)

            tm_11_basis = tf.get_variable(name="tm_11_basis", initializer=tf.constant(diag_init, dtype=tf.float32))
            tm_12_basis = tf.get_variable(name="tm_12_basis", initializer=tf.constant(tm12_init, dtype=tf.float32))
            tm_21_basis = tf.get_variable(name="tm_21_basis", initializer=tf.constant(tm21_init, dtype=tf.float32))
            tm_22_basis = tf.get_variable(name="tm_22_basis", initializer=tf.constant(diag_init, dtype=tf.float32))
            self._basises = [tm_11_basis, tm_12_basis, tm_21_basis, tm_22_basis]

            if self.c.transition_network_hidden_dict is not None:
                tm_hidden = NDenseHiddenLayers(params=self.c.transition_network_hidden_dict, name_prefix="tmHidden")
            else:
                tm_hidden = None
            tm_out = FactorizedLLRKNOutputLayerBasis("tm", num_basises=self.c.num_basis, basises=self._basises)
            self._tm_net = FeedForwardNet(tm_out, hidden_layers=tm_hidden)

            if self.c.learn_state_dependent_transition_covar:
                if self.c.transition_covar_hidden_dict is not None:
                    tc_hidden = NDenseHiddenLayers(params=self.c.transition_covar_hidden_dict, name_prefix='tcHidden')
                else:
                    tc_hidden = None
                tc_out = SimpleOutputLayer(name_prefix= "tcOut",
                                           output_dim=2 * self.latent_observation_dim,
                                           activation=self.c.variance_activation_fn)
                self._tc_net = FeedForwardNet(tc_out, hidden_layers=tc_hidden)
                def net(x):
                    transition_matrix = self._tm_net(x)
                    transition_covar = self._tc_net(x)
                    return transition_matrix + [transition_covar[:, :self.latent_observation_dim],
                                                transition_covar[:, self.latent_observation_dim:]]
            else:
                net =self._tm_net

            self.transition_cell = FactorizedLLTransitionCell(config=self.c, network=net,
                                                              transition_covar=self._transition_covar,
                                                              state_dependent_variance=self.c.learn_state_dependent_transition_covar,
                                                              debug=self.debug_recurrent)


class FactorizedLLRKNOutputLayerBasis(OutputLayer):

    def __init__(self, name_prefix, num_basises, basises):
        self._coeff_out = tf.layers.Dense(name=name_prefix + "Out", units=num_basises, activation=tf.nn.softmax)
        self._basises = basises

    def __call__(self, last_hidden):

        coefficients = tf.expand_dims(self._coeff_out(last_hidden), -1)
        return [tf.reduce_sum(coefficients * x, 1) for x in self._basises]

class FactorizedLLRKNOutputLayer(OutputLayer):

    def __init__(self, latent_observation_dim, name_prefix, with_variance=True, init_lim=1e-5, var_activation=None):
        """ Create new output layer
        :param output_dim: dimensionality of output
        :param name_prefix: name of the layer
        """

        if with_variance:
            assert var_activation is not None, "FactorizedLLLRKNOutputLayer with variance requested but no variance" \
                                               "activation function given!"

        super().__init__(output_dim=latent_observation_dim * (6 if with_variance else 4),
                         name_prefix=name_prefix)

        #self._latent_obs_dim = latent_observation_dim

        #With this initialization the network should output matrices close to the identity matrix at the beginning of
        # training, yielding a "stable" system
        tm_11_layer_raw = tf.layers.Dense(units=latent_observation_dim,
                                          activation=tf.identity,
                                          kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                          bias_initializer=tf.initializers.zeros())
        tm_11_layer = lambda x: tm_11_layer_raw(x) + tf.ones([1, latent_observation_dim])
        tm_12_layer_raw = tf.layers.Dense(units=latent_observation_dim,
                                          activation=tf.identity,
                                          kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                          bias_initializer=tf.initializers.zeros())
        tm_12_layer = lambda x: tm_12_layer_raw(x)
        tm_21_layer_raw = tf.layers.Dense(units=latent_observation_dim,
                                          activation=tf.identity,
                                          kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                          bias_initializer=tf.initializers.zeros())
        tm_21_layer = lambda x: tm_21_layer_raw(x)
        tm_22_layer_raw  = tf.layers.Dense(units=latent_observation_dim,
                                           activation=tf.identity,
                                           kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                           bias_initializer=tf.initializers.zeros())
        tm_22_layer = lambda x: tm_22_layer_raw(x) + tf.ones([1, latent_observation_dim])
        self._layers = [tm_11_layer, tm_12_layer, tm_21_layer, tm_22_layer]

        if with_variance:
            tc_u_layer = tf.layers.Dense(units=latent_observation_dim,
                                         activation=var_activation,
                                         kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                         bias_initializer=tf.initializers.constant(np.log(0.001)))
            tc_l_layer = tf.layers.Dense(units=latent_observation_dim,
                                         activation=var_activation,
                                         kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                         bias_initializer=tf.initializers.constant(np.log(0.001)))
            self._layers += [tc_u_layer, tc_l_layer]

    def __call__(self, last_hidden):
        return [l(last_hidden) for l in self._layers]

    @property
    def output_dim(self):
        return self._output_dim