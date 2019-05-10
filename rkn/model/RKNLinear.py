import tensorflow as tf
from transition_cell.TransitionCell import TransitionCell
from transition_cell.RKNFullTransitionCell import RKNFullTransitionCell
from transition_cell.RKNCorrTransitionCell import RKNCorrTransitionCell
from transition_cell.RKNSimpleTransitionCell import RKNSimpleTransitionCell
import util.LossFunctions as loss_fn
import numpy as np
from util.MulUtil import bmatmul

class RKNLinear:


    def __init__(self,
                 config,
                 observation_covar=None,
                 observation_covar_init=None,
                 debug_recurrent=False,
                 use_lstm=False):
        self.c = config
        self.state_dim = self.c.latent_state_dim
        self.observation_dim = self.c.latent_observation_dim
        self.transition_cell_type = self.c.transition_cell_type
        self.debug_recurrent = debug_recurrent
        self.use_lstm = use_lstm

        if self.transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
            self._covar_tile_suffix = [1]
            self.TransitionCell = RKNSimpleTransitionCell
        elif self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
            self._covar_tile_suffix = [1, 1]
            self.TransitionCell = RKNFullTransitionCell
        elif self.transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
            self._covar_tile_suffix = [1]
            self.TransitionCell = RKNCorrTransitionCell

        self._build_transition_matrix()
        self._build_transition_noise_covar()
        self._build_initial_covar()
        self._build_observation_covar(observation_covar, observation_covar_init)
        self._build_inputs()
        self._build_model()
        self._build_optimizer()
        self.tf_session = tf.Session()
        self.tf_session.run(tf.global_variables_initializer())


    def _build_transition_matrix(self):
        if self.c.transition_matrix_is_given:
            self.transition_matrix = tf.constant(self.c.transition_matrix, dtype=tf.float32)
        else:
            transition_matrix_init = np.eye(self.state_dim) + np.random.uniform(low=-0.1, high=0.1, size=[self.state_dim, self.state_dim])
            assert not np.any(transition_matrix_init == 0), "Invalid inital transition matrix sampled"
            self.transition_matrix = tf.get_variable(name="transition_matrix",
                                                     dtype=tf.float32,
              #                                       shape=[self.state_dim, self.state_dim],
              #                                       initializer=tf.glorot_uniform_initializer())
                                                     initializer=tf.constant(transition_matrix_init, dtype=tf.float32))

    def _build_transition_noise_covar(self):
        if self.c.transition_covariance_given:
            self._transition_covar = tf.constant(self.c.transition_covariance, dtype=tf.float32)
        else:
            init_upper, init_lower = self.c.transition_covariance_init[:2]

            log_transition_covar_upper = tf.get_variable(name="log_transition_covar_upper",
                                                         dtype=tf.float32,
                                                         initializer=tf.constant([np.log(init_upper)]))
            self.transition_covar_upper = self.c.variance_activation_fn(log_transition_covar_upper)

            log_transition_covar_lower = tf.get_variable(name="log_transition_covar_lower",
                                                         dtype=tf.float32,
                                                         initializer=tf.constant([np.log(init_lower)]))
            self.transition_covar_lower = self.c.variance_activation_fn(log_transition_covar_lower)
            covars = [self.transition_covar_upper, self.transition_covar_lower]
            if not self.transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
                init_side = self.c.transition_covariance_init[2]
                self.transition_covar_side =  tf.constant([0], dtype=tf.float32)#tf.get_variable(name="log_transition_covar_side",
                                             #               dtype=tf.float32,
                                             #               initializer=tf.constant([init_side]))
                covars += [self.transition_covar_side]
            if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
                full_covar_upper, full_covar_lower, full_covar_side = \
                    [tf.matrix_diag(tf.tile(x, [self.observation_dim])) for x in covars]
                self._transition_covar = tf.concat([tf.concat([full_covar_upper, full_covar_side], 1),
                                                    tf.concat([full_covar_side, full_covar_lower], 1)],0)
            else:
                self._transition_covar = tf.concat([tf.tile(x, [self.observation_dim]) for x in covars], 0)

    def _build_initial_covar(self):
        if self.c.initial_state_covariance_given:
            self._initial_state_covar = tf.constant(self.c.initial_state_covariance, dtype=tf.float32)
        else:
            log_initial_state_covar = tf.get_variable(name="log_initial_state_covar",
                                                      dtype=tf.float32,
                                                      initializer=tf.constant([np.log(self.c.initial_state_covariance_init)]))
            self.initial_state_covar = self.c.variance_activation_fn(log_initial_state_covar)
            self._initial_state_covar = tf.tile(self.initial_state_covar, [self.state_dim])

            if self.transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
                self._initial_state_covar = tf.concat([self._initial_state_covar, tf.zeros(self.observation_dim)], 0)
            elif self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
                self._initial_state_covar = tf.matrix_diag(self._initial_state_covar)

    def _build_observation_covar(self, observation_covar=None, init=None):
        assert observation_covar is None or init is None, "Both observation covar and initial value for learning it given"
        assert observation_covar is not None or init is not None, "Neither, observation covar nor initial value for learning it given"
        if observation_covar is not None:
            self._observation_covar = tf.constant(observation_covar, dtype=tf.float32)
        else:
            log_observation_covar = tf.get_variable(name="log_observation_covar",
                                                    dtype=tf.float32,
                                                    initializer=tf.constant([np.log(init)]))
            self.observation_covar = self.c.variance_activation_fn(log_observation_covar)
            self._observation_covar = tf.tile(self.observation_covar, [self.observation_dim])

            if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
                self._observation_covar = tf.matrix_diag(self._observation_covar)

    def _build_inputs(self):
        self.observations = tf.placeholder(dtype=tf.float32,
                                           shape=[None, None, self.observation_dim],
                                           name="observations_in")
        self.targets = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, self.observation_dim],
                                      name="targets_in")

    def _build_model(self):
        batch_size = tf.shape(self.observations)[0]
        seq_len = tf.shape(self.observations)[1]

        self.transition_cell = self.TransitionCell(config=self.c,
                                                   transition_matrix=self.transition_matrix,
                                                   transition_covar=self._transition_covar,
                                                   debug=self.debug_recurrent)

        obs_covar_ext = tf.expand_dims(tf.expand_dims(self._observation_covar, 0), 0)
        obs_covar_full = tf.tile(obs_covar_ext, [batch_size, seq_len] + self._covar_tile_suffix)
        self.inputs = self.transition_cell.pack_input(self.observations, obs_covar_full, tf.ones([batch_size, seq_len, 1]))

        initial_mean = tf.zeros([batch_size, self.state_dim])
        initial_covar_ext = tf.expand_dims(self._initial_state_covar, 0)
        initial_covar_full = tf.tile(initial_covar_ext, [batch_size] + self._covar_tile_suffix)
        self.initial_states = self.transition_cell.pack_state(initial_mean, initial_covar_full)

       # lstm_state_dim = 10
       # lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_state_dim)

        self.full_predicted_latent_states, self.last_state = \
            tf.nn.dynamic_rnn(cell=self.transition_cell,
                              inputs=self.inputs,
                              initial_state=self.initial_states)
       # h1 = tf.layers.dense(self.full_predicted_latent_states, units=10)
       # decoded = tf.layers.dense(h1, units=4)
     #   decoder_mat = tf.get_variable("DecoderMat", shape=[lstm_state_dim, 2 * self.state_dim], dtype=tf.float32,
     #                                 initializer=tf.glorot_uniform_initializer())
     #   decoded = bmatmul(self.full_predicted_latent_states, decoder_mat)

        self.predicted_mean, self.predicted_covar = \
            self.transition_cell.unpack_state(self.full_predicted_latent_states)

        #self.predicted_covar = tf.exp(self.predicted_covar)

    def _build_optimizer(self):
        predicted_obs_mean = self.predicted_mean[:, :, :self.observation_dim]
        if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
            predicted_obs_var = tf.matrix_diag_part(self.predicted_covar[:, :, :self.observation_dim, :self.observation_dim])
        else:
            predicted_obs_var = self.predicted_covar[:, :, :self.observation_dim]
        if self.c.use_likelihood:
            if self.observation_dim > 1 and self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
                tf.logging.warn("gaussian nll currently assumes diagonal")
            self.training_loss = loss_fn.gaussian_nll(self.targets, predicted_obs_mean,  predicted_obs_var)
            self.reference_loss = loss_fn.rmse(self.targets, predicted_obs_mean)
        else:
            self.training_loss = loss_fn.rmse(self.targets, predicted_obs_mean)
            self.reference_loss = self.training_loss
        self.loss = [self.training_loss, self.reference_loss]
        if len(tf.trainable_variables()) > 0:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.c.learning_rate).minimize(self.training_loss)

    def _feed_inputs(self, observations, targets=None):

        feed_dict = {self.observations: observations}
        if targets is not None:
            feed_dict[self.targets] = targets
        return feed_dict

    def train(self, observations, targets, training_epochs):
        num_seq = observations.shape[0]
        seq_len = observations.shape[1]
        if num_seq % self.c.batch_size != 0:
            tf.logging.warn("Amount of sequences not divided by batch_size - ignoring a few elements each epoch")
        if seq_len % self.c.bptt_length != 0:
            tf.logging.warn("Sequence length not divided by bptt_length - last part shorter")
        num_of_batches = int(np.floor(num_seq / self.c.batch_size))
        seq_parts = int(np.ceil(seq_len / self.c.bptt_length))
        for epoch in range(training_epochs):
            avg_loss = 0
            suffled_idx = np.random.permutation(num_seq)
            for i in range(num_of_batches):
                batch_slice = suffled_idx[i * self.c.batch_size : (i + 1) * self.c.batch_size]
                for j in range(seq_parts):
                    seq_slice = slice(j * self.c.bptt_length, (j  + 1) * self.c.bptt_length)
                    feed_dict = self._feed_inputs(observations[batch_slice, seq_slice],
                                                  targets[batch_slice, seq_slice])
                    _, loss, last_latent = self.tf_session.run(fetches=[self.optimizer, self.loss, self.last_state],
                                                               feed_dict=feed_dict)
                    avg_loss += np.array(loss) / (num_of_batches * seq_parts)
                    if np.any(np.isnan(avg_loss)):
                        self.c.mprint("Loss = NaN - abort")
                        return
            self.c.mprint('Epoch', (epoch + 1), ': Loss', avg_loss)

    def evaluate(self, observations, targets):
        loss = self.tf_session.run(fetches=self.loss, feed_dict=self._feed_inputs(observations, targets))
        self.c.mprint('Evaluation Loss: ', loss)

    def filter(self, observations):
        return self.tf_session.run(fetches=(self.predicted_mean, self.predicted_covar),
                                   feed_dict=self._feed_inputs(observations))

    def eval_tensor(self, tensor):
        return tensor.eval(session=self.tf_session)
