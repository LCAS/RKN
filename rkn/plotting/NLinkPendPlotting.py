import numpy as np

from plotting import Common as p
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt


class NLinkPendulumPlotter:
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
        # Noise Factors between 0 and 1
        self.color_range = [0, 1]

    def plot_position_sequences(self, predictions, targets, noise_fact=None):
        """Plots predicted and true positions
        :param predictions: predicted values
        :param targets: target values
        :param noise_fact: amount of added noise (relative, between 0 and 1)
        """
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(1, true=targets[i, :, :],
                                        pred=predictions[i, :, :],
                                        labels=["Position"],
                                        colors=noise_fact[i, :] if noise_fact is not None else None,
                                        color_range=self.color_range)
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_predictions' + str(i))

    def plot_predictions(self, predictions, targets, variance=None):
        if variance is not None:
            target_in = np.concatenate([targets, variance], axis=-1)
        else:
            target_in = targets

        labels = ["sin link 1", "cos link 1",
                  "sin link 2", "cos link 2",
                  "sin link 3", "cos link 3",
                  "sin link 4", "cos link 4"
                  "var sin link 1", "var cos link 1",
                  "var sin link 2", "var cos link 2",
                  "var sin link 3", "var cos link 3",
                  "var sin link 4", "var cos link 4"
                  ]

        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(target_in.shape[-1],
                                        true=target_in[i],
                                        pred=predictions[i],
                                  #      labels=labels,
                                        variance=variance[i] if variance is not None else variance)
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_predictions' + str(i))


    def plot_variance(self, variance, noise_facts=None):
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(variance.shape[-1], variance[i], colors=noise_facts[i] if noise_facts is not None else None, color_range=[0, 1])
            p.save_fig(fig = fig, path=self.plot_path, name=self.file_name_prefix + 'variance' + str(i))

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

    def plot_diagnostics(self, obs_mean, obs_covar,
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
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'Diagnostic' + str(i))

    def plot_observation_sequences(self, targets, predictions, inputs):
        sequence_length = predictions.shape[1]
        for i in range(self.plot_n):
            fig = p.plot_pictures_sequence(true=np.squeeze(targets[i]),
                                           input=np.squeeze(inputs[i]),
                                           prediction=np.squeeze(predictions[i, :, :, -1:0:-1, :]),
                                           color=False,
                                           n=sequence_length)
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'image_sequences' + str(i))

    def close_all(self):
        plt.close("all")
