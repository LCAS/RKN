import tensorflow as tf
import e2c.E2C_Util as u
import numpy as np
from network.FeedForwardNet import FeedForwardNet
from network.OutputLayer import GaussianOutputLayer, UpConvOutputLayer
from network.HiddenLayers import NDenseHiddenLayers,NConvolutionalHiddenLayers, HiddenLayerWrapper, ReshapeLayer
from util import MeanCovarPacker as mcp

class E2CTransitionModel:

    def __init__(self, state_dim, action_dim, hidden_dict):
        self.init_func = tf.glorot_uniform_initializer()
        self.activation = tf.nn.relu
        self.hidden_dict = hidden_dict
        self.state_dim = state_dim
        self.action_dim = action_dim

    def transition(self, current_mean, current_action, current_sig):
        with tf.name_scope('transition_model'):
            self._hidden_layers = NDenseHiddenLayers(self.hidden_dict, "transition_hidden")
            self.layer_v = tf.layers.Dense(self.state_dim, tf.identity, kernel_initializer=self.init_func)
            self.layer_r = tf.layers.Dense(self.state_dim, tf.identity, kernel_initializer=self.init_func)
            self.layer_bflat = tf.layers.Dense(self.state_dim * self.action_dim, tf.identity,
                                               kernel_initializer=self.init_func)
            self.layer_o = tf.layers.Dense(self.state_dim, tf.identity, kernel_initializer=self.init_func)

            h_2 = self._hidden_layers(current_mean, True)
            v = self.layer_v(h_2)
            r = self.layer_r(h_2)
           # b_flat = self.layer_bflat(h_2)
            o = self.layer_o(h_2)

            vr = tf.matmul(tf.expand_dims(v, 2), tf.expand_dims(r, 2), transpose_b=True, name="vr")
            A = vr + tf.diag(tf.ones([self.state_dim]))
            Az = tf.matmul(A, tf.expand_dims(current_mean, 2))

            #B = tf.reshape(b_flat, [-1, self.state_dim, self.action_dim])
            #Bu = tf.matmul(B, tf.expand_dims(current_action, 2))

            #next_mean = tf.reshape(Az + Bu + tf.expand_dims(o, 2), [-1, self.state_dim])
            next_mean = tf.reshape(Az + tf.expand_dims(o, 2), [-1, self.state_dim])
            a_diag = tf.matrix_diag_part(A)
            next_sig = current_sig * tf.square(a_diag)
            #return next_mean, next_sig, A, B, o
            return next_mean, next_sig, A, o

class E2C:

    def __init__(self,
                 observation_dim,
                 action_dim,
                 latent_dim,
                 learning_rate,
                 factor_regularization_loss,
                 factor_distance_loss,
                 encoder_conv_dict,
                 encoder_dense_dict,
                 decoder_dense_dict,
                 decoder_conv_dict,
                 decoder_init_shape,
                 transition_dict,
                 use_gpu,
                 reconstruction_loss_function=u.binary_crossentropy):

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.factor_regularization_loss = factor_regularization_loss
        self.factor_distance_loss = factor_distance_loss
        self.use_gpu = use_gpu
        self._build_encoder(encoder_dense_dict, encoder_conv_dict)
        self._build_decoder(decoder_dense_dict, decoder_conv_dict, decoder_init_shape)
        self.transition_dict = transition_dict
        self.reconstruction_loss_function = reconstruction_loss_function

        self.tf_session = tf.InteractiveSession()
        self._build_model()
        self._build_optimizer()

        self.tf_session.run(tf.initialize_all_variables())

    def _build_encoder(self, encoder_dense_dict, encoder_conv_dict):
        """ Builds encoder network """
        encoder_hidden_dense = NDenseHiddenLayers(params=encoder_dense_dict,
                                                  name_prefix='EncoderHiddenDense')
        encoder_hidden_conv = NConvolutionalHiddenLayers(params=encoder_conv_dict,
                                                         name_prefix='EncoderHiddenConv',
                                                         on_gpu=self.use_gpu,
                                                         flatten_output=True,
                                                         up_convolutional=False)
        encoder_hidden = HiddenLayerWrapper([encoder_hidden_conv, encoder_hidden_dense])

        encoder_out = GaussianOutputLayer(distribution_dim=self.latent_dim,
                                          name_prefix='EncoderOut',
                                          single_variance=False,
                                          max_variance=-1,
                                          variance_fn=tf.exp,
                                          constant_variance=False,
                                          fix_constant_variance=False,
                                          variance=np.nan,
                                          normalize_mean=False)

        self.encoder = FeedForwardNet(output_layer=encoder_out,
                                      hidden_layers=encoder_hidden)

    def _build_decoder(self, decoder_dense_dict, decoder_conv_dict, decoder_initial_shape):
        """ Builds decoder network """
        decoder_hidden_dense = NDenseHiddenLayers(params=decoder_dense_dict,
                                                      name_prefix='DecoderHiddenDense')
        decoder_hidden_reshape = ReshapeLayer(shape=decoder_initial_shape,
                                              name_prefix="DecoderHiddenReshape",
                                              on_gpu=self.use_gpu)
        decoder_hidden_conv = NConvolutionalHiddenLayers(params=decoder_conv_dict,
                                                         name_prefix='DecoderHiddenUpConv',
                                                         on_gpu=self.use_gpu,
                                                         flatten_output=False,
                                                         up_convolutional=True)
        decoder_hidden = HiddenLayerWrapper([decoder_hidden_dense, decoder_hidden_reshape, decoder_hidden_conv])
        decoder_out = UpConvOutputLayer(output_dim=self.observation_dim,
                                        name_prefix='DecoderOut',
                                        on_gpu=self.use_gpu,
                                        activation=tf.nn.sigmoid)
        self.decoder = FeedForwardNet(output_layer=decoder_out, hidden_layers=decoder_hidden)

    def _build_model(self):
        with tf.name_scope('Model'):

            self.x_t0 = tf.placeholder(tf.float32, [None] + self.observation_dim, name="x_t0")
            self.x_t1 = tf.placeholder(tf.float32, [None] + self.observation_dim, name="x_t1")
#            self.x_t0_target = tf.placeholder(tf.float32, [None] + [self.observation_dim], name="x_t0_target")
#            self.x_t1_target = tf.placeholder(tf.float32, [None] + [self.observation_dim], name="x_t1_target")
            self.u = tf.placeholder(tf.float32, [None, self.action_dim], name="u")
            self.training = tf.placeholder(tf.bool, [], name='training')

            self.transition_model = E2CTransitionModel(state_dim=self.latent_dim, action_dim=self.action_dim,
                                                       hidden_dict=self.transition_dict)

            # encode and decode x_t0
            with tf.variable_scope('encoder'):
                self.z_t0_mean, self.z_t0_sig = mcp.unpack(self.encoder(self.x_t0, self.training), self.latent_dim)
                self.z_t0 = u.reparameterize_normal(self.z_t0_mean, self.z_t0_sig, self.training)


            with tf.variable_scope('decoder'):
                self.x_t0_reconst_mean = self.decoder(self.z_t0)

            with tf.variable_scope('transition'):
                self.z_hat_mean, self.z_hat_sig, _, _ \
                    = self.transition_model.transition(self.z_t0_mean,
                                                       self.u,
                                                       self.z_t0_sig)
                self.z_t1_hat = u.reparameterize_normal(self.z_hat_mean, self.z_hat_sig, self.training)

            with tf.variable_scope('decoder', reuse=True):
                self.x_t1_reconst_mean = self.decoder(self.z_t1_hat)

            with tf.variable_scope('encoder', reuse=True):
                self.z_t1_mean, self.z_t1_sig = mcp.unpack(self.encoder(self.x_t1, self.training), self.latent_dim)
                self.z_t1 = u.reparameterize_normal(self.z_t1_mean, self.z_t1_sig, self.training)

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.reconstruction_loss_x_t0 = self.reconstruction_loss_function(self.x_t0, self.x_t0_reconst_mean)
            self.reconstruction_loss_x_t1 = self.reconstruction_loss_function(self.x_t1, self.x_t1_reconst_mean)
            self.regularization_loss = self.factor_regularization_loss * u.gaussian_kl(self.z_t0_mean,
                                                                                       self.z_t0_sig)
            self.distance_loss = self.factor_distance_loss * u.gaussian_kl(self.z_hat_mean, self.z_hat_sig,
                                                                           self.z_t1_mean,  self.z_t1_sig)

            self.loss = self.reconstruction_loss_x_t0 + self.reconstruction_loss_x_t1 +\
                        self.regularization_loss + self.distance_loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def train(self, obs_t0, obs_t1, batch_size, training_epochs=10):
        num_of_batches = int(len(obs_t0) / batch_size)
        for train_epoch in range(training_epochs):
            avg_loss = avg_r0_loss = avg_r1_loss = avg_reg_loss = avg_dist_loss = 0
            rnd_idx = np.random.permutation(len(obs_t0))
            for i in range(num_of_batches):
                cur_slice = slice(i * batch_size, (i + 1) * batch_size)
                batch_xs = obs_t0[rnd_idx[cur_slice]]
                batch_ys = obs_t1[rnd_idx[cur_slice]]
                batch_us = np.zeros([batch_size, self.action_dim])
                loss, r0_loss, r1_loss, l_loss, d_loss = self._train_on_batch(batch_xs, batch_ys, batch_us)
                avg_loss += loss / num_of_batches
                avg_r0_loss += r0_loss / num_of_batches
                avg_r1_loss += r1_loss / num_of_batches
                avg_reg_loss += l_loss / num_of_batches
                avg_dist_loss += d_loss / num_of_batches
            print("Epoch:", '%04d' % (train_epoch + 1),
                  "loss=", "{:.9f}".format(avg_loss),
                  "reconstruction (t)=", "{:.9f}".format(avg_r0_loss),
                  "reconstruction (t+1)=", "{:.9f}".format(avg_r1_loss),
                  "regularization=", "{:.9f}".format(avg_reg_loss),
                  "distance=", "{:.9f}".format(avg_dist_loss))

    def _train_on_batch(self, X, Y, u):
        opt, loss, reconstruction_loss_0, reconstruction_loss_1, regularization_loss, distance_loss = \
            self.tf_session.run((self.optimizer, self.loss,
                                 self.reconstruction_loss_x_t0, self.reconstruction_loss_x_t1,
                                 self.regularization_loss, self.distance_loss),
                                feed_dict={self.x_t0: X, self.x_t1: Y, self.u: u, self.training: True})
        return loss, reconstruction_loss_0, reconstruction_loss_1, regularization_loss, distance_loss

    def evaluate(self, obs_t0, obs_t1, test_batch_size=-1):
        num_of_batches = 1 if test_batch_size < 0 else int(len(obs_t0) / test_batch_size)
        test_batch_size = len(obs_t0) if test_batch_size < 0 else test_batch_size
        predictions = np.zeros(obs_t0.shape)
        avg_loss = avg_r0_loss = avg_r1_loss = avg_reg_loss = avg_dist_loss = 0
        for i in range(num_of_batches):
            current_slice = slice(i * test_batch_size, (i+1) * test_batch_size)
            current_prediction, loss,  r0_loss, r1_loss, reg_loss, d_loss =\
                self.tf_session.run((self.x_t1_reconst_mean,
                                     self.loss, self.reconstruction_loss_x_t0, self.reconstruction_loss_x_t1,
                                     self.regularization_loss, self.distance_loss),
                                    feed_dict={self.x_t0: obs_t0[current_slice],
                                               self.u: np.zeros([test_batch_size, self.action_dim]),
                                               self.x_t1: obs_t1[current_slice],
                                               self.training: False})
            predictions[current_slice, :] = current_prediction
            avg_loss += loss / num_of_batches
            avg_r0_loss += r0_loss / num_of_batches
            avg_r1_loss += r1_loss / num_of_batches
            avg_reg_loss += reg_loss / num_of_batches
            avg_dist_loss += d_loss / num_of_batches
        print("Evaluation:",
              "loss=", "{:.9f}".format(avg_loss),
              "reconstruction (t)=", "{:.9f}".format(avg_r0_loss),
              "reconstruction (t+1)=", "{:.9f}".format(avg_r1_loss),
              "regularization=", "{:.9f}".format(avg_reg_loss),
              "distance=", "{:.9f}".format(avg_dist_loss))

        return predictions, avg_loss, avg_r0_loss, avg_r1_loss, avg_reg_loss, avg_dist_loss


    def bce(self, targets, predictions):
        targets = targets.astype(np.float64) / 255.0
        epsilon = 1e-8
        point_wise_error \
            = - (targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
        sample_wise_error = np.sum(point_wise_error, axis=(-3, -2, -1))
        return np.mean(sample_wise_error)

    def impute_sequence(self, obs, obs_valid, targets, img_size):
        imputed_obs = np.zeros(obs.shape)
        reconstructed_obs = np.zeros(obs.shape)
        z = np.zeros([1, self.latent_dim])
        for i in range(len(obs)):
            if obs_valid[i]:
                z = self.tf_session.run(self.z_t0_mean, feed_dict={self.x_t0: obs[i:i+1],
                                                                   self.training: False})
                imputed_obs[i] = obs[i].astype(imputed_obs.dtype) / 255.0
                reconstructed_obs[i] = self.tf_session.run(fetches=self.x_t0_reconst_mean, feed_dict={self.z_t0: z})
            else:
                z = self.tf_session.run(fetches=self.z_hat_mean, feed_dict={self.z_t0_mean: z})
                imputed_obs[i] = self.tf_session.run(fetches=self.x_t0_reconst_mean, feed_dict={self.z_t0: z})
                reconstructed_obs[i] = imputed_obs[i]
      #  seq_loss = self.bce(targets, imputed_obs)
        imputed_loss = self.bce(targets[:, -img_size:], imputed_obs[:, -img_size:])
        reconstruction_loss = self.bce(targets[:, -img_size:], reconstructed_obs[:, -img_size:])
        return imputed_loss, reconstruction_loss, reconstructed_obs




