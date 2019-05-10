import tensorflow as tf
from model.RKN import RKN
import numpy as np
from network.OutputLayer import OutputLayer, SimpleOutputLayer
from network.HiddenLayers import  NDenseHiddenLayers
from network.FeedForwardNet import FeedForwardNet
from transition_cell.FFLLTransitionCell import FFLLTransitionCell

MODE_UNCONSTRAINT = "unconst"
MODE_FIX_UPPER = "fix_upper"
MODE_FIX_UPPER_VELO = "fix_upper_velo"

class FFLLRKN(RKN):

    def _build_transition_cell(self):
        self.c.mprint("Using really Fully Factorized LL-RKN Transition Cell")
        with tf.name_scope("TransitionModel"):
            self.transition_nets = []
            for i in range(self.latent_observation_dim):
                h = NDenseHiddenLayers(params=self.c.transition_network_hidden_dict,
                                       name_prefix="TransitionHidden" + str(i))
                out = FFLLRKNOutputLayer(name_prefix="TransitionOut"+ str(i),
                                         #mode=MODE_UNCONSTRAINT,
                                         #mode=MODE_FIX_UPPER_VELO,
                                         #mode=MODE_FIX_UPPER,
                                         with_variance=self.c.learn_state_dependent_transition_covar,
                                         init_lim=1e-5, var_activation=self.c.variance_activation_fn)
                self.transition_nets.append(FeedForwardNet(out, h))

            self.transition_cell = FFLLTransitionCell(self.c, networks=self.transition_nets, debug=self.debug_recurrent,
                                                      state_dependent_variance=self.c.learn_state_dependent_transition_covar)


class FFLLRKNBasis(RKN):

    def _build_transition_cell(self):
        mode=MODE_UNCONSTRAINT
        #mode = MODE_FIX_UPPER_VELO
        #mode=MODE_FIX_UPPER

        if mode == MODE_UNCONSTRAINT:
            base_init = np.concatenate([np.ones([self.c.num_basis, 1]), np.random.rand(self.c.num_basis, 1) * 0.2,
                                        -np.random.rand(self.c.num_basis, 1) * 0.2, np.ones([self.c.num_basis, 1])], 1)
        elif (mode == MODE_FIX_UPPER) or (mode == MODE_FIX_UPPER_VELO):
            base_init = np.concatenate([-np.random.rand(self.c.num_basis, 1) * 0.2, np.ones([self.c.num_basis, 1])], 1)
        base_var = tf.get_variable("base_var_uncostraint", initializer=tf.constant(base_init, dtype=tf.float32))
        base = tf.expand_dims(base_var, 0)


        self.c.mprint("Using really Fully Factorized LL-RKN Transition Cell")
        with tf.name_scope("TransitionModel"):
            self.transition_nets = []
            for i in range(self.latent_observation_dim):
                if self.c.transition_network_hidden_dict is not None:
                    h = NDenseHiddenLayers(params=self.c.transition_network_hidden_dict,
                                           name_prefix="TransitionHidden" + str(i))
                else:
                    h = None
                out = FFLLRKNOutputLayerBasis(name_prefix="TransitionOut"+ str(i),
                                              mode=mode,
                                              basises=base,
                                              with_variance=self.c.learn_state_dependent_transition_covar,
                                              init_lim=1e-5, var_activation=self.c.variance_activation_fn)
                self.transition_nets.append(FeedForwardNet(out, h))

            self.transition_cell = FFLLTransitionCell(self.c, networks=self.transition_nets, debug=self.debug_recurrent,
                                                      transition_covar=self._transition_covar,
                                                      state_dependent_variance=self.c.learn_state_dependent_transition_covar)

class FFLLRKNOutputLayer(OutputLayer):

    def __init__(self, name_prefix, mode, with_variance=True, init_lim=1e-5, var_activation=None):
        """ Create new output layer
        :param output_dim: dimensionality of output
        :param name_prefix: name of the layer
        """

        if with_variance:
            assert var_activation is not None, "FactorizedLLLRKNOutputLayer with variance requested but no variance" \
                                               "activation function given!"

        output_dim = 2
        output_dim += (2 if mode == MODE_UNCONSTRAINT else 0)

        super().__init__(output_dim=(6 if with_variance else 4),
                         name_prefix=name_prefix)
        self._with_variance = with_variance

        tm_out_layer_raw = tf.layers.Dense(units=output_dim,
                                           activation=tf.identity,
                                           kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                           bias_initializer=tf.initializers.zeros())

        def fix_upper_layer(x):
            upper_row = tf.concat([tf.ones([tf.shape(x)[0], 1]), 0.1 * tf.ones([tf.shape(x)[0], 1])], -1)
            return tf.concat([upper_row, tm_out_layer_raw(x) + tf.constant([[0.0, 1.0]])], -1)

        if mode == MODE_UNCONSTRAINT:
            print("MODE UNCONSTRAINT")

            self.tm_out_layer = lambda x: tm_out_layer_raw(x) + tf.constant([[1.0, 0.0, 0.0, 1.0]])
        elif mode == MODE_FIX_UPPER:
            print("MODE FIX UPPER")

            self.tm_out_layer = fix_upper_layer
        elif mode == MODE_FIX_UPPER_VELO:
            print("MODE VELO")
            def fix_upper_layer_velo(x):
                out = fix_upper_layer(x)
                return tf.concat([out[:, :3], tf.nn.sigmoid(out[:, 3:4])], -1)
            self.tm_out_layer = fix_upper_layer_velo

        if self._with_variance:
            self.tc_out_layer = tf.layers.Dense(units=2,
                                                activation=var_activation,
                                                kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                                bias_initializer=tf.initializers.zeros())

    def __call__(self, last_hidden):
        if self._with_variance:
            return [self.tm_out_layer(last_hidden), self.tc_out_layer(last_hidden)]
        else:
            return self.tm_out_layer(last_hidden)

    @property
    def output_dim(self):
        return self._output_dim

class FFLLRKNOutputLayerBasis(OutputLayer):

    def __init__(self, name_prefix, mode, basises, with_variance=True, init_lim=1e-5, var_activation=None):
        """ Create new output layer
        :param output_dim: dimensionality of output
        :param name_prefix: name of the layer
        """

        num_basises = basises.shape[1]
        self._mode = mode
        self._basises = basises

        if with_variance:
            assert var_activation is not None, "FactorizedLLLRKNOutputLayer with variance requested but no variance" \
                                               "activation function given!"

        super().__init__(output_dim=(6 if with_variance else 4),
                         name_prefix=name_prefix)
        self._with_variance = with_variance

        self.tm_coeff_layer = tf.layers.Dense(units=num_basises,
                                              activation=tf.nn.softmax)

        if self._with_variance:
            self.tc_out_layer = tf.layers.Dense(units=2,
                                                activation=var_activation,
                                                kernel_initializer=tf.initializers.random_uniform(-init_lim, init_lim),
                                                bias_initializer=tf.initializers.zeros())

    def __call__(self, last_hidden):
        coeffs = tf.expand_dims(self.tm_coeff_layer(last_hidden), -1)

        if self._mode == MODE_UNCONSTRAINT:
            print("MODE UNCONSTRAINT")
            tm = tf.reduce_sum(coeffs * self._basises, 1)

        elif self._mode == MODE_FIX_UPPER or self._mode == MODE_FIX_UPPER_VELO:
            upper = tf.tile(tf.constant([[1.0 , 0.1]]), [tf.shape(last_hidden)[0], 1])
            lower = tf.reduce_sum(coeffs * self._basises, 1)
            if self._mode == MODE_FIX_UPPER:
                print("MODE FIX UPPER")
                tm = tf.concat([upper, lower], -1)
            elif self._mode == MODE_FIX_UPPER_VELO:
                print("MODE VELO")
                lower = tf.stack([lower[:, 0], tf.nn.sigmoid(lower[:, 1])], 1)
                tm = tf.concat([upper, lower], -1)
        if self._with_variance:
            return [tm, self.tc_out_layer(last_hidden)]
        else:
            return tm

    @property
    def output_dim(self):
        return self._output_dim