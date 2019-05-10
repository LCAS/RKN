from plotting import Common as p
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt

class TBTPlotter:
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

    def plot_variance(self, variance, noise_facts=None):
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(variance.shape[-1], variance[i], colors=noise_facts[i] if noise_facts is not None else None, color_range=[0, 1])
            p.save_fig(fig = fig, path=self.plot_path, name=self.file_name_prefix + 'variance' + str(i))


    def plot_predictions(self, predictions, targets):
        labels = ["Box 1 x", "Box 1 y", "Box 1 z", "Box 2 x", "Box 2 y", "Box 2 z", "Box 3 x", "Box 3 y", "Box 3 z"]
        for i in range(self.plot_n):
            fig = p.plot_n_trajectories(targets.shape[-1],
                                        true=targets[i],
                                        pred=predictions[i],
                                        labels=labels)
            p.save_fig(fig, self.plot_path, self.file_name_prefix + '_predictions' + str(i))

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


    def close_all(self):
        plt.close("all")