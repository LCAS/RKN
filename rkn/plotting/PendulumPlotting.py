from scipy.stats import norm
import numpy as np

from plotting import Common as p
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt


class PendulumPlotter:
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

    def plot_predictions(self, predictions, targets, colors=None, variance=None):
        if variance is not None:
            target_in = np.concatenate([targets, variance], axis=-1)
        else:
            target_in = targets

        if predictions.shape[-1] == 2:
            labels = ["Position X", "Position Y", "Variance X", "Variance Y"]
        else:
            labels = ["sin 1", "cos 1", "sin 2", "cos 2", "sin 3", "cos 3",
                      "sigma sin 1", "sigma cos 1", "sigma sin 2", "sigma cos 2", "sigma sin 3", "sigma cos 3"]

        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(target_in.shape[-1],
                                        true=target_in[i],
                                        pred=predictions[i],
                                        labels=labels,
                                        colors=colors[i] if colors is not None else None,
                                        variance=variance[i] if variance is not None else None) #["Position X", "Position Y", "Variance X", "Variance Y"])
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_predictions' + str(i))

    def plot_noise_factors(self, noise_factors):
        """Plots the factors of the added noise
        :param noise_factors:
        """
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(1, noise_factors[i, :, np.newaxis], labels=['Noise Factor'])
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'noise_factors' + str(i))

    def plot_observation_sequences(self, targets, predictions, inputs):
        sequence_length = predictions.shape[1]
        for i in range(self.plot_n):
            fig = p.plot_pictures_sequence(true=np.squeeze(targets[i]),
                                           input=np.squeeze(inputs[i]),
                                           prediction=np.squeeze(predictions[i]),
                                           color=False,
                                           n=sequence_length)
            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'image_sequences' + str(i))

    def plot_variance(self, variance, noise_facts=None):
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(variance.shape[-1], variance[i], colors=noise_facts[i] if noise_facts is not None else None, color_range=[0, 1])
            p.save_fig(fig = fig, path=self.plot_path, name=self.file_name_prefix + 'variance' + str(i))

    def plot_smoothing(self, target, true, pred):
        for i in range(self.plot_n):
            fig = plt.figure()

            ax = plt.subplot(2, 1, 1)
            plt.plot(target[i, :, 0], c="blue")
            plt.plot(true[i, :, 0], c= "green")
            plt.plot(pred[i, :, 0], c="red")
            plt.legend(["targets", "ground_truth", "prediction"])

            ax = plt.subplot(2, 1, 2)
            plt.plot(target[i, :, 1], c="blue")
            plt.plot(true[i, :, 1], c= "green")
            plt.plot(pred[i, :, 1], c="red")

            p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'smoothing' + str(i))


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

    def plot_histogram(self, targets, predictions, variance):
        fig = plt.figure()
        if len(targets.shape) == 5:
            variance = np.reshape(variance, [variance.shape[0], variance.shape[1], 1, 1, 1])
        s = (targets - predictions) / np.sqrt(variance)
        plt.hist(np.ravel(s), bins=100, normed=True, range=[-4, 4])
        x_plt = np.arange(-4, 4, 0.01)
        plt.plot(x_plt, norm.pdf(x_plt))
        p.save_fig(fig=fig, path=self.plot_path, name=self.file_name_prefix + 'hist')


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

    def close_all(self):
        plt.close("all")
