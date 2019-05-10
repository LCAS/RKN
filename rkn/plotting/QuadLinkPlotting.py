import numpy as np

from plotting import Common as p


class QuadLinkPlotter:
    """Plots Results of pendulum experiments"""
    def __init__(self, plot_path, file_name_prefix, plot_n=10):
        """
        :param plot_path: path to save the images under - if not existent it will be created
        :param file_name_prefix: prefix for all images created
        :param plot_n: number of results to be plotted
        """
        self.plot_path = plot_path
        self.file_name_prefix = file_name_prefix
        self.plot_n = plot_n

        self.diagnostic_labels = ['Latent Observation Covariance Norm',
                                  'Latent State Covariance Norm',
                                  'Latent State Norm',
                                  'Latent State Max Value',
                                  'Latent State Min Value']
        # Noise Factors between 0 and 1
        self.color_range = [0, 1]

    def plot_position_sequences(self, predictions, inputs, targets):
        """Plots predicted and true positions
        :param predictions: predicted values
        :param targets: target values
        :param noise_fact: amount of added noise (relative, between 0 and 1)
        """
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories_tripple(2, true=targets[i, :, :],
                                                input=inputs[i, :, :],
                                                pred=predictions[i, :, :],
                                                labels=["Position X", "Position Y"])
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_predictions' + str(i))

    def plot_variance(self, variance):
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(variance.shape[-1], variance[i])
            p.save_fig(fig = fig, path=self.plot_path, name=self.file_name_prefix + 'variance' + str(i))



    def plot_diagnostics(self, full_latent_states, full_latent_observations):
        """Plots diagnostic information about the latent observation and state"""
        latent_states, latent_state_covar = p.unpack_mean_covar(full_latent_states)
        latent_observations, latent_observation_covar = p.unpack_mean_covar(full_latent_observations)


        latent_observation_covar_norm = \
            np.linalg.norm(latent_observation_covar, axis=(-1), keepdims=True)
        latent_state_covar_norm = np.linalg.norm(latent_state_covar, axis=(-1), keepdims=True)

        rest = latent_state_covar_norm.shape[1] - latent_observation_covar_norm.shape[1]
        if rest > 0:
            padding = np.zeros([latent_observation_covar_norm.shape[0], rest, 1])
            latent_observation_covar_norm = np.concatenate([latent_observation_covar_norm, padding], 1)

        latent_state_norm = np.linalg.norm(latent_states, axis=(-1), keepdims=True)

        latent_state_max_val = np.max(np.abs(latent_states), axis=-1, keepdims=True)
        latent_state_min_val = np.min(np.abs(latent_states), axis=-1, keepdims=True)

        diagonstics_list = [latent_observation_covar_norm,
                            latent_state_covar_norm,
                            latent_state_norm,
                            latent_state_max_val,
                            latent_state_min_val]
        diagnositcs = np.concatenate(diagonstics_list, axis=-1)

        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(5,
                                        diagnositcs[i, :, :],
                                        labels=self.diagnostic_labels,
                                        colors=None,
                                        color_range=self.color_range)
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'Diagnostic' + str(i))
