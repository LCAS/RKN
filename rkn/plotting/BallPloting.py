import numpy as np

from plotting import Common as p
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt



class BallPlotter:

    def __init__(self, plot_path, file_name_prefix, plot_n=3):
        self.plot_path = plot_path
        self.file_name_prefix = file_name_prefix
        self.plot_n = plot_n

        self.diagnostic_labels = ['Latent Observation Covariance Norm',
                                  'Latent State Covariance Norm',
                                  'Latent State Norm',
                                  'Latent State Max Value',
                                  'Latent State Min Value']
        # Noise Factors between 0 and 1

    def plot_position_sequences(self, predictions, targets, visibility, n_balls):
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(2, true=targets[i, :, :],
                                        pred=predictions[i, :, :],
                                        labels=["Position X", "Position Y"],
                                        colors=visibility[i, :, np.newaxis],
                                        color_range=[0, np.max(visibility)])
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_' + str(n_balls) +'_predictions' + str(i))

    def plot_predictions(self, n_balls, predictions, targets, variance=None, visibility=None):
        if variance is not None:
            target_in = np.concatenate([targets, variance], axis=-1)
        else:
            target_in = targets

        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(target_in.shape[-1],
                                        true=target_in[i],
                                        pred=predictions[i],
                                        colors=visibility[i],
                                        color_range=[0, 21],
                                        labels=["Position X", "Position Y", "Variance X", "Variance Y"])
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_' + str(n_balls) + '_predictions' + str(i))


    def plot_visibility(self, visibility, n_balls):
        """Plots the factors of the added noise
        :param :
        """
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(1, visibility[i, :, np.newaxis], labels=['Visibility'])
            p.save_fig(fig=fig, path=self.plot_path,
                       name=self.file_name_prefix + '_' + str(n_balls) + '_noise_factors' + str(i))

    def plot_transition_matrix(self, mat):
        fig = p.plot_transition_matrix(mat)
        p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'transitionMat')

    def plot_loss_per_time(self, targets, predictions, variance=None):
        if variance is not None:
            nll_per_time = p.nll_time_step_wise(targets, predictions, variance)
            n_plots = 2
        else:
            n_plots = 1
        reference_loss_per_time = p.reference_loss_per_time(targets, predictions)

        fig = plt.figure()
        ct = 1
        if variance is not None:
            ax = plt.subplot(n_plots, 1, ct)
            plt.plot(nll_per_time)
            plt.title("nll per time")
            ct += 1

        ax = plt.subplot(n_plots, 1, ct)
        plt.plot(reference_loss_per_time)
        plt.title("rmse per time")

        p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'loss_per_time')

    def plot_diagnostics(self, n_balls, obs_mean, obs_covar,
                         post_mean, post_covar,
                         prior_mean=None, prior_covar=None,
                         transition_covar=None, kalman_gain=None,
                         noise_factors=None):
        """Plots diagnostic information about the latent observation and state"""

        for i in range(self.plot_n):
            fig = p.plot_diagnostics(i=i, obs_mean=obs_mean, obs_covar=obs_covar,
                                     post_mean=post_mean, post_covar=post_covar,
                                     prior_mean=prior_mean, prior_covar=prior_covar,
                                     transition_covar=transition_covar, kalman_gain=kalman_gain,
                                     noise_factors=noise_factors)
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + '_' + str(n_balls) + '_Diagnostic' + str(i))

    def close_all(self):
        plt.close("all")
