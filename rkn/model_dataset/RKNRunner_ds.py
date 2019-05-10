from model.RKNRunner import RKNRunner
from model_dataset.RKN_ds import RKN_ds
import numpy as np

class RKNRunner_ds(RKNRunner):

    def __init__(self, model):

        assert isinstance(model, RKN_ds), "RKNRunner_ds only works with RKN_ds, if you use RKN use RKNRunner"
        super().__init__(model)

    def train(self,
              training_epochs,
              num_batches,
              verbose_interval=1):
        #This need to go inside the loop!
        self.tf_session.run(self.model.train_iterator.initializer)
        for epoch in range(training_epochs):
            avg_loss = 0
            for j in range(num_batches):
                feed_dict = self._feed_inputs()
                feed_dict[self.model.training] = True
                loss, _ = self._train_step(feed_dict)
            avg_loss += loss / num_batches

            if np.any(np.isnan(avg_loss)):
                print("Loss = NaN - abort")
                return None

            if (epoch + 1) % verbose_interval == 0:
                print('Epoch', (epoch + 1), ': Loss', avg_loss)




    def _feed_inputs(self, observations=None, targets=None, initial_latent_state=None, observation_valid=None):
        """Creates the feed - handles the defaults for 'initial_latent_state' and 'observation_valid'
        :param observations: Observations to feed in
        :param targets: Targets to feed in
        :param initial_latent_state: Initial latent states to feed in
        :param observation_valid: Observation valid flags to feed in
        """

        if initial_latent_state is None:
            initial_latent_state = np.empty(shape=[observations.shape[0], 2 * self.model.latent_state_dim])
            initial_latent_state[:] = np.NAN

        if observation_valid is None:
            obs_shape = np.shape(observations)
            observation_valid = np.ones([obs_shape[0], obs_shape[1], 1], dtype=np.bool)


        feed_dict = {self.model.observations_valid: observation_valid,
                     self.model.initial_latent_state: initial_latent_state}

        if observations is None:
            feed_dict[self.model.use_ph] = False
        else:
            feed_dict[self.model.use_ph] = True
            feed_dict[self.model.observations_ph] = observations
            if targets is not None:
                feed_dict[self.model.targets_ph] = targets

        return feed_dict
