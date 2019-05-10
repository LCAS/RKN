import numpy as np
import matplotlib.pyplot as plt
import pykalman
class SpringPendulum:

    def __init__(self, m=1, k=1, b=0.5, dt=0.1, dim=1, transition_covar=None, observation_covar=None, seed=0):
        assert dim == 1 or dim == 2, "Invalid Dimensionality 'dim', only 1 and 2 supported"

        np.random.seed(seed)

        self.k = k
        self.b = b
        self.m = m
        self.dt = dt
        self.obs_dim = dim
        self.state_dim = dim * 2


        self.transition_matrix =  np.eye(self.state_dim) + self.dt * self._get_a()

        self.observation_matrix = np.concatenate([np.eye(self.obs_dim), np.zeros([self.obs_dim, self.obs_dim])], 1)

        if transition_covar is None:
            self.transition_covar = (0.05**2) * np.eye(self.state_dim)
        elif np.isscalar(transition_covar):
            self.transition_covar = transition_covar * np.eye(self.state_dim)
        else:
            self.transition_covar = transition_covar

        if observation_covar is None:
            self.observation_covar = (0.1**2) * np.eye(self.obs_dim)
        elif np.isscalar(observation_covar):
            self.observation_covar = observation_covar * np.eye(self.obs_dim)
        else:
            self.observation_covar = observation_covar

        self.gt_kf = pykalman.KalmanFilter(transition_matrices=self.transition_matrix,
                                           observation_matrices=self.observation_matrix,
                                           transition_covariance=self.transition_covar,
                                           observation_covariance=self.observation_covar)

    def _get_a(self):
        if self.state_dim == 2:
            return np.array([[0,                    1],
                             [- self.k / self.m, - self.b / self.m]])
        elif self.state_dim == 4:
            return np.array([[0,                0,                 1,                0               ],
                             [0,                0,                 0,                1               ],
                             [-self.k / self.m, 0,                 -self.b / self.m, 0               ],
                             [0,                - self.k / self.m, 0,                -self.b / self.m]])



    def sample_sequences(self, num_sequences, sequence_length):
        states = np.zeros([num_sequences, sequence_length, self.state_dim])
        states[:, 0] = np.random.rand(num_sequences, self.state_dim) * 2 - 1
        for i in  range(sequence_length - 1):
            transition_noise = np.random.multivariate_normal(mean=np.zeros(self.state_dim), cov=self.transition_covar, size=num_sequences)
            states[:, i+1] = np.matmul(states[:, i], self.transition_matrix.T) + transition_noise

        observation_noise = np.random.multivariate_normal(mean=np.zeros(self.obs_dim), cov=self.observation_covar,
                                                          size=[num_sequences, sequence_length])
        observations = states[:, :, 0:self.obs_dim] + observation_noise

        return states, observations

    def get_kf_gt(self, observations):
        num_seq = observations.shape[0]
        seq_len = observations.shape[1]
        filtered_mean = np.zeros([num_seq, seq_len, self.state_dim])
        filtered_covars = np.zeros([num_seq, seq_len, self.state_dim, self.state_dim])
        for i in range(num_seq):
            cur_filtered_mean, cur_filtered_covars = self.gt_kf.filter(observations[i])
            filtered_mean[i] = cur_filtered_mean
            filtered_covars[i] = cur_filtered_covars
        return filtered_mean, filtered_covars

    def generate_images(self, observations, dim=200):
        min_val = np.min(observations)
        max_val = np.max(observations)
        pixels = np.floor((dim-1) * (observations - min_val) / (max_val - min_val)).astype(np.int32)
        images = np.zeros([observations.shape[0], observations.shape[1], dim])
        for i in range(observations.shape[0]):
            for j in range(observations.shape[1]):
                images[i, j, pixels[i, j]] = 1
        return images


if __name__ == '__main__':
    dim = 1

    spring_pend = SpringPendulum(m=-1, k=-0.1, b=0, dt=-0.1, dim=dim)

    states, observations = spring_pend.sample_sequences(1, 100)
    kf_mean, kf_covar = spring_pend.get_kf_gt(observations)

    if dim == 1:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(observations[0, :, 0], c='red')
        plt.plot(states[0, :, 0], c='blue')
        plt.plot(kf_mean[0, :, 0], c='green')


        plt.subplot(2, 1, 2)
        plt.plot(states[0, :, 1], c='blue')
        plt.plot(kf_mean[0, :, 1], c='green')
    elif dim == 2:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(observations[0, :, 0], c='red')
        plt.plot(states[0, :, 0], c='blue')
        plt.plot(kf_mean[0, :, 0], c='green')


        plt.subplot(4, 1, 2)
        plt.plot(observations[0, :, 1], c='red')
        plt.plot(states[0, :, 1], c='blue')
        plt.plot(kf_mean[0, :, 1], c='green')

        plt.subplot(4, 1, 3)
        plt.plot(states[0, :, 2], c='blue')
        plt.plot(kf_mean[0, :, 2], c='green')


        plt.subplot(4, 1, 4)
        plt.plot(states[0, :, 3], c='blue')
        plt.plot(kf_mean[0, :, 3], c='green')

    plt.show()
    #images = spring_pend.generate_images(observations)


  #  plt.figure()
  #  plt.imshow(images[0].T, cmap="gray", interpolation="None")
  #  plt.show()
