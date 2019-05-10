import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt

import os
import warnings
import numpy as np


def plot_transition_matrix(mat):
    fig = plt.figure()
    c = plt.imshow(mat, interpolation="none")
    fig.colorbar(c)
    return fig


def plot_n_trajectories(n, true, pred=None, colors=None, labels=None, ylim=None, color_range=[0, 89], variance=None):
    """
    Plots n different trajectories in one image
    :param n: number of trajectories to plot
    :param true: true trajectory
    :param pred: plotted trajectory
    :param colors: colors of scatter plot
    :param labels: Added as headline above the plot
    :param ylim: Limits of y axes
    :param color_range: Interval the values in colors could be in.
    :return:
    """

    two_std = 2 * np.sqrt(variance)
    fig = plt.figure(figsize=[4, 2*n])

    for i in range(n):
        ax = plt.subplot(n, 1, i + 1)

        plt.plot(true[:, i])
        if colors is not None:
            c = colors[..., int(i/ 2) % colors.shape[-1]]
            sc = plt.scatter(np.arange(len(true)), true[:, i], marker='+', linewidths=2, c=c, cmap=plt.cm.autumn)

        if pred is not None and i < pred.shape[-1]:
            if variance is not None:
                plt.plot(pred[:, i], color='green')
                plt.fill_between(np.arange(len(pred[:, i])), pred[:, i] - two_std[:, i], pred[:, i] + two_std[:, i],
                                 alpha=0.5, edgecolor='green', facecolor='green')
            else:
                plt.plot(pred[:, i], color='green')
        if labels is not None:
            plt.title(labels[i])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim([-5, np.shape(true)[0] + 5])

    if colors is not None:
        cbar = fig.colorbar(sc, ticks=color_range, orientation='horizontal')
        plt.clim(*[color_range])

        cbar.ax.set_xticklabels(['covered', 'observable'])
    fig.tight_layout()
    return fig

def plot_n_trajectories_tripple(n, true, input, pred=None, labels=None, ylim=None, color_range=[0, 89]):
    """
    Plots n different trajectories in one image
    :param n: number of trajectories to plot
    :param true: true trajectory
    :param pred: plotted trajectory
    :param colors: colors of scatter plot
    :param labels: Added as headline above the plot
    :param ylim: Limits of y axes
    :param color_range: Interval the values in colors could be in.
    :return:
    """

    fig = plt.figure(figsize=[4, 2*n])

    for i in range(n):
        ax = plt.subplot(n, 1, i + 1)
        plt.plot(true[:, i])

        if input is not None and i < pred.shape[-1]:
            plt.plot(input[:, i], color='red')
        if pred is not None and i < pred.shape[-1]:
            plt.plot(pred[:, i], color='green')
        if labels is not None:
            plt.title(labels[i])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim([-5, np.shape(true)[0] + 5])
    fig.tight_layout()
    return fig

def plot_diagnostics(i, obs_mean, obs_covar,
                     post_mean, post_covar,
                     prior_mean=None, prior_covar=None,
                     transition_covar=None, kalman_gain=None,
                     noise_factors=None,
                     color_range=None):
    """Plots diagnostic information about the latent observation and state"""

    latent_obs_dim = obs_mean.shape[-1]
    episode_len = post_mean.shape[1]

    num_plots = 4 + int(post_covar.shape[-1] / latent_obs_dim)
    num_plots += 2 if prior_mean is not None else 0
    num_plots += int(prior_covar.shape[-1] / latent_obs_dim) if prior_covar is not None else 0
    num_plots += int(transition_covar.shape[-1] / latent_obs_dim) if transition_covar is not None else 0
    num_plots += int(kalman_gain.shape[-1] / latent_obs_dim) if kalman_gain is not None else 0

    ct = 1
    fig = plt.figure(figsize=[4, num_plots * 2])

    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(obs_mean[i, :, :])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Latent Observations")

    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(obs_covar[i, :, :])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Latent Observation Covar")
    if noise_factors is not None:
        plt.scatter(np.arange(np.shape(obs_covar)[1]), obs_covar[i, :, 0], marker="+", linewidths=2, c=noise_factors[i, :], cmap=plt.cm.autumn)


    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(post_mean[i, :, :latent_obs_dim])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Posterior Latent State Upper")

    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(post_mean[i, :, latent_obs_dim:])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Posterior Latent State Lower")

    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(post_covar[i, :, :latent_obs_dim])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Posterior Latent State Covar Upper")

    ax = plt.subplot(num_plots, 1, ct)
    ct += 1
    plt.plot(post_covar[i, :, latent_obs_dim: 2 * latent_obs_dim])
    ax.set_xlim([-5, episode_len + 5])
    plt.title("Posterior Latent State Covar Lower")

    if post_covar.shape[-1] / latent_obs_dim > 2:
        ax = plt.subplot(num_plots, 1, ct)
        ct +=1
        plt.plot(post_covar[i, :, 2 * latent_obs_dim:])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Posterior Latent State Covar Side")

    if prior_mean is not None:
        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(prior_mean[i, :, :latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Prior Latent State Mean Upper")

        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(prior_mean[i, :, latent_obs_dim:])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Prior Latent State Mean Lower")

    if prior_covar is not None:
        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(prior_covar[i, :, :latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Prior Latent State Covar Upper")

        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(prior_covar[i, :, latent_obs_dim: 2 * latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Prior Latent State Covar Lower")

        if prior_covar.shape[-1] / latent_obs_dim > 2:
            ax = plt.subplot(num_plots, 1, ct)
            ct += 1
            plt.plot(prior_covar[i, :, 2 * latent_obs_dim:])
            ax.set_xlim([-5, episode_len + 5])
            plt.title("Prior Latent State Covar Side")

    if transition_covar is not None:
        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(transition_covar[i, :, :latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Transition Covar Upper")

        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(transition_covar[i, :, latent_obs_dim: 2 * latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Transition Covar Lower")

        if transition_covar.shape[-1] / latent_obs_dim > 2:
            ax = plt.subplot(num_plots, 1, ct)
            ct += 1
            plt.plot(transition_covar[i, :, 2 * latent_obs_dim:])
            ax.set_xlim([-5, episode_len + 5])
            plt.title("Transition Covar Side")

    if kalman_gain is not None:
        ax = plt.subplot(num_plots, 1, ct)
        ct += 1
        plt.plot(kalman_gain[i, :, :latent_obs_dim])
        ax.set_xlim([-5, episode_len + 5])
        plt.title("Kalman Gain Upper")
        if kalman_gain.shape[-1] /latent_obs_dim > 1:
            ax = plt.subplot(num_plots, 1, ct)
            ct += 1
            plt.plot(kalman_gain[i, :, latent_obs_dim:])
            ax.set_xlim([-5, episode_len + 5])
            plt.title("Kalman Gain Lower")

    fig.tight_layout()
    return fig

def plot_pictures_sequence(true, input=None, prediction=None, n=20, offset=0, color=False):
    def plt_img(img, i, column):
        ax = plt.subplot(n, plot_width, plot_width * i + column)
        plt.imshow(img, interpolation='none')
        if not color:
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plot_width = 1
    if input is not None:
        plot_width += 1
    if prediction is not None:
        plot_width += 1
    fig = plt.figure(figsize=(plot_width, n))

    for i in range(0, n):
        plt_img(true[offset + i], i, 1)
        if input is not None and offset + i < np.shape(input)[0]:
            plt_img(input[offset + i], i, 2)
        if prediction is not None:
            plt_img(prediction[offset + i], i, 3)
    return fig

def save_fig(fig, path, name, dpi=1200):
    """ Saves figure as pdf
    :param fig: Figure to be saved
    :param path: path to save figure under - created if non existent
    :param name: filename to save figure under
    :param dpi: resolution
    :return:
    """
    if not os.path.exists(path):
        warnings.warn('Path ' + path + ' not found - creating')
        os.makedirs(path)
    mp.rcParams['lines.linewidth'] = .2
    fig.savefig(os.path.join(path, name + '.pdf'), format='pdf', dpi=dpi)
    plt.close(fig)

def unpack_mean_covar(vector):
    """ extracts mean and covariance vector from concatenated vector
    :param vector:
    :return:
    """
    dim = int(np.shape(vector)[-1] / 2)
    return vector[:, :, :dim], vector[:, :, dim:]

def nll_time_step_wise(targets, predictions, variance):
    if len(variance.shape) < len(predictions.shape):
        variance = np.expand_dims(np.expand_dims(variance, -1), -1)
        reduction_dims  = (-3, -2, -1)
    else:
        reduction_dims = -1
    const = np.log(2 * np.pi)
    element_wise_nll = 0.5 * (const + np.log(variance) + ((targets - predictions)**2) / variance)
    sample_wise_nll = np.sum(element_wise_nll, reduction_dims)
    return np.mean(sample_wise_nll, 0)

def reference_loss_per_time(targets, predicitons):
    if len(targets.shape) == 5:
        return bce_per_time(targets, predicitons)
    else:
        return rmse_per_time(targets, predicitons)

def bce_per_time(targets, predicitons):
    point_wise_error = - (targets * np.log(predicitons) + (1 - targets) * np.log(1-predicitons))
    sample_wise_error = np.sum(point_wise_error, (-3, -2, -1))
    return np.mean(sample_wise_error)



def rmse_per_time(targets, predictions):
    return np.sqrt(np.mean((targets - predictions)**2, (0, 2)))