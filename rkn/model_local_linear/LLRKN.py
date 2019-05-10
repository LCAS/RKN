from transition_cell.LLRKNTransitionCellCorr import LLRKNTransitionCell
from model.RKN import RKN
from network import NDenseHiddenLayers, SimpleOutputLayer, FeedForwardNet
import tensorflow as tf
import numpy as np

class LLRKN(RKN):


    def _build_transition_matrix(self):
        with tf.name_scope("TransitionModel"):
            if self.c.transition_matrix_init == RKN.TRANS_INIT_FIX:
                tm_11_init = tf.eye(self.latent_observation_dim, batch_shape=[self.c.num_basis])
                tm_12_init = 0.2 * tf.eye(self.latent_observation_dim, batch_shape=[self.c.num_basis])
                tm_21_init = -0.2 * tf.eye(self.latent_observation_dim, batch_shape=[self.c.num_basis])
                tm_22_init = tf.eye(self.latent_observation_dim, batch_shape=[self.c.num_basis])
                tm_11_full = tf.get_variable(name="tm_11_basis", initializer=tm_11_init)
                tm_12_full = tf.get_variable(name="tm_12_basis", initializer=tm_12_init)
                tm_21_full = tf.get_variable(name="tm_21_basis", initializer=tm_21_init)
                tm_22_full = tf.get_variable(name="tm_22_basis", initializer=tm_22_init)
                tm_11, tm_12, tm_21, tm_22 = (tf.matrix_band_part(x, self.c.bandwidth, self.c.bandwidth) for x in
                                              [tm_11_full, tm_12_full, tm_21_full, tm_22_full])
                self.transition_matrix_basis = tf.concat([tf.concat([tm_11, tm_12], -1),
                                                          tf.concat([tm_21, tm_22], -1)], -2)
            else:
                raise NotImplementedError("Not Implemented")

            if self.c.transition_network_hidden_dict is not None:
                transition_hidden = NDenseHiddenLayers(params=self.c.transition_network_hidden_dict,
                                                       name_prefix='TransitionHidden')
            else:
                transition_hidden = None
            transition_out = SimpleOutputLayer(output_dim=self.c.num_basis,
                                               name_prefix='TransitionOut',
                                               activation=tf.nn.softmax)
            self._transition_net = FeedForwardNet(transition_out, transition_hidden)

    def _build_transition_cell(self):
        """ Builds transition model
        config.num_basis = 1 => equivalent to RKN (correlated), i.e. "the one from the nips paper"
        config.bandwidth = 0 => equivalent to LLRKNFactorized
        config.bandwidth = config.latent_observation_dim => equivalent to LLRKN
        config.bandwidth > config.latent_observation_dim => error
         """
        self.c.mprint("using LL-RKN Transition Cell")



                #transition_init = np.eye(self.latent_state_dim)
                #transition_init = np.tile(np.expand_dims(transition_init, 0), [self.c.num_basis, 1, 1])
                #for i in range(self.c.num_basis):
                #    transition_init[i, self.latent_observation_dim:, :self.latent_observation_dim] = - 0.2 * np.random.rand()  * np.eye(self.latent_observation_dim)
                #    transition_init[i, :self.latent_observation_dim, self.latent_observation_dim:] =  0.2 * np.random.rand() * np.eye(self.latent_observation_dim)


                #self.transition_matrix_basis = tf.get_variable(name="transition_matrix_basis",
                #                                               initializer=tf.constant(transition_init, dtype=tf.float32))


            #else:
             #   transition_net = lambda x, training: tf.ones(shape=[1])

        self.transition_covar = self._transition_covar
        self.transition_cell = LLRKNTransitionCell(config=self.c,
                                                   basis_matrices=self.transition_matrix_basis,
                                                   coefficenet_fn=lambda x: self._transition_net(x, self.training),
                                                   debug=self.debug_recurrent,
                                                   transition_covar=self._transition_covar)



