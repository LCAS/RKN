from model.RKN import RKN
import tensorflow as tf

class RKN_ds(RKN):

    def __init__(self, config, train_iterator, test_iterator):
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        super().__init__(config)

    def _build_inputs(self):
        self.use_ph = tf.placeholder_with_default(tf.constant(False), shape=[])
        self.training = tf.placeholder(tf.bool, shape=[])

        self.observations_ph = tf.placeholder(dtype=tf.float32,
                                           shape=[None, None] + self.input_dim,
                                           name="observations")

        output_dim = self.output_dim if isinstance(self.output_dim, list) else [self.output_dim]
        self.targets_ph = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None] + output_dim,
                                      name='targets')

        self.observations_valid = tf.placeholder(dtype=tf.bool,
                                                 shape=[None, None, 1],
                                                 name="observations_valid")

        self.initial_latent_state = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, 2 * self.latent_state_dim],
                                                   name="initial_latent_state")

        train_observations_ds, train_targets_ds = self.train_iterator.get_next()
        test_observations_ds, test_targets_ds = self.test_iterator.get_next()

        test_observations, test_targets = tf.cond(self.use_ph,
                                                  lambda: (self.observations_ph, self.targets_ph),
                                                  lambda: (test_observations_ds, test_targets_ds))

        self.observations, self.targets = tf.cond(self.training,
                                                  lambda:(train_observations_ds, train_targets_ds),
                                                  lambda:(test_observations, test_targets))


