from model.RKNRunner import RKNRunner
import numpy as np

class GRUBaselineRunner(RKNRunner):

    def _feed_inputs(self, observations, targets=None, initial_latent_state=None, observation_valid=None):
        """Creates the feed - handles the defaults for 'initial_latent_state' and 'observation_valid'
        :param observations: Observations to feed in
        :param targets: Targets to feed in
        :param initial_latent_state: Initial latent states to feed in
        :param observation_valid: Observation valid flags to feed in
        """
        if observation_valid is None:
            obs_shape = np.shape(observations)
            observation_valid = np.ones([obs_shape[0], obs_shape[1], 1], dtype=np.bool)

        init_dummy = np.zeros(shape=[observations.shape[0], 2 * self.model.latent_state_dim])

        feed_dict = {self.model.observations: observations,
                     self.model.observations_valid: observation_valid,
                     self.model.initial_latent_state: init_dummy}

        if targets is not None:
            feed_dict[self.model.targets] = targets

        return feed_dict