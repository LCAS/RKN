import tensorflow as tf
import abc

class TransitionCell(tf.nn.rnn_cell.RNNCell):

    TRANSITION_CELL_SIMPLE = "TransitionCellSimple"
    TRANSITION_CELL_FULL = "TransitionCellFull"
    TRANSITION_CELL_CORRELATED = "TransitionCellCorr"

    """Implementation of the Transition Layer described in the paper. This is implemented as a subclass of the
    tensorflow cell and can hence used with tf.nn.dynamic_rnn"""

    def __init__(self,
                 config,
                 transition_matrix,
                 transition_covar,
                 diag_only=False,
                 debug=False):
        self.c = config
        self.diag_only = diag_only
        self.transition_matrix = transition_matrix
        self.transition_covar = transition_covar
        self.latent_state_dim = self.c.latent_state_dim
        self.latent_observation_dim = self.c.latent_observation_dim

        self.debug = debug
        assert self.latent_state_dim == 2 * self.latent_observation_dim, "Currently only 2 * m = n supported"

    def __call__(self, inputs, states ,scope=None):
        """Performs one transition step (prediction followed by update in Kalman Filter terms)
        Parameter names match those of superclass
        :param inputs: Latent Observations (mean and covariance vectors concatenated)
        :param states: Last Latent State (mean and covariance vectors concatenated)
        :param scope: See super
        :return: Current Latent State (mean and covariance vectors concatenated)
        """
        with tf.name_scope("PrepareInputs"):
            observation_mean, observation_covar, observation_valid = self.unpack_input(inputs)
            state_mean, state_covar = self.unpack_state(states)
        return self._transition(state_mean, state_covar, observation_mean, observation_covar, observation_valid)


    def _transition(self, state_mean, state_covar, observation_mean, observation_covar, observation_valid):
        """Performs transition step if last latent state is given. Assumes input state to be normalized!
        :param state_mean: last latent state mean
        :param state_covar: last latent state covariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent observation covariance
        :param observation_valid: indicating if observation is valid
        :return: Next state
        """
        with tf.name_scope('Transition'):
            pred_res = self._predict(state_mean, state_covar)
            prior_state_mean, prior_state_covariance = pred_res[:2]

            self.c.mprint("normalizing prior:", end="")
            prior_state_mean, prior_state_covariance = self._normalize_if_desired(prior_state_mean, prior_state_covariance, self.c.normalize_prior)
            prior_state = self.pack_state(prior_state_mean, tf.concat(prior_state_covariance, -1))

            #self.c.mprint("normalizing observations:", end="")
            #observation_mean, observation_covar = self._normalize_if_desired(observation_mean, observation_covar, self.c.normalize_obs)

            if self.c.never_invalid:
                update_res = self._update(prior_state_mean, prior_state_covariance,
                                          observation_mean, observation_covar)
            else:
                update_res = self._masked_update(prior_state_mean, prior_state_covariance,
                                                 observation_mean, observation_covar, observation_valid)



            post_state_mean, post_state_covariance = update_res[:2]
            self.c.mprint("normalizing posterior:", end="")
            post_state_mean, post_state_covariance = self._normalize_if_desired(post_state_mean, post_state_covariance, self.c.normalize_posterior)
            post_state = self.pack_state(post_state_mean, tf.concat(post_state_covariance, -1))
            if self.c.n_step_pred > 0:
                pred_state_mean = post_state_mean
                pred_state_covariance = post_state_covariance
                self.c.mprint("Running", self.c.n_step_pred, "step prediction")
                for i in range(self.c.n_step_pred):
                    pred_res = self._predict(pred_state_mean, pred_state_covariance)
                    pred_state_mean, pred_state_covariance = self._normalize_if_desired(*pred_res[:2], self.c.normalize_prior)
                pred_state = self.pack_state(pred_state_mean, pred_state_covariance)
                cell_out = tf.concat([pred_state, prior_state])
            else:
                cell_out = tf.concat([post_state, prior_state], -1)
            if self.debug:
                pred_debug = pred_res[2] * tf.ones([tf.shape(post_state)[0], 3 * self.latent_observation_dim])
                debug_out = tf.concat([cell_out, pred_debug, update_res[2]], -1)
                return debug_out, post_state
            else:
                return cell_out, post_state

    @abc.abstractmethod
    def _predict(self, state_mean, state_covar):
        """ Performs prediction step
        :param state_mean: last posterior mean
        :param state_covar: last posterior covariance
        :return: current prior latent state mean and covariance
        """
        raise NotImplementedError("Not Implemented")

    def _masked_update(self, state_mean, state_covar, observation_mean, observation_covar, observation_valid):
        """ Ensures update only happens if observation is valid
        :param state_mean: current prior latent state mean
        :param state_covar: current prior latent state convariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent observation covariance
        :param observation_valid: indicating if observation is valid
        :return: current posterior latent state mean and covariance
        """
        # TODO Reimplement without cond, not supported on gpu
        a = tf.concat(self._update(state_mean, state_covar, observation_mean, observation_covar), -1)
        b = tf.concat((state_mean, state_covar, tf.zeros(tf.shape(observation_mean))) if self.debug else (state_mean, state_covar), -1)
        x = tf.where(tf.expand_dims(observation_valid, -1), a, b)
        return x


        #all_valid = tf.reduce_all(observation_valid)
        #return tf.cond(all_valid,
        #               lambda: self._update(state_mean, state_covar, observation_mean, observation_covar),
        #               lambda: ((state_mean, state_covar, tf.zeros(tf.shape(observation_mean))) if self.debug else (
        #               state_mean, state_covar)))

    @abc.abstractmethod
    def _update(self, state_mean, state_covar, observation_mean, observation_covar):
        """Performs update step
        :param state_mean: current prior latent state mean
        :param state_covar: current prior latent state covariance
        :param observation_mean: current latent observation mean
        :param observation_covar: current latent covariance mean
        :return: current posterior latent state and covariance
        """
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def pack_state(self, mean, covar):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def unpack_state(self, state_as_vector):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def pack_input(self, observation_mean, observation_covar, observation_valid):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def unpack_input(self, input_as_vector):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def _normalize_if_desired(self, mean, covar, flag):
        return NotImplementedError("Not Implemented")
