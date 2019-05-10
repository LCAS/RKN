import numpy as np
from data.AbstractBalls import AbstractBalls

class LinearBalls(AbstractBalls):
    STATE_DIM = 4

    def __init__(self,
                 n_balls,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 dyn_sigma=0.0,
                 b=0.99,
                 m=100,
                 dt=1,
                 max_init_vel=0.1,
                 seed=None):
        """
        Creates new linear balls object, also see super
        :param b and m: transition model parameters
        :param dt: difference between time steps
        :param max_init_vel: maximal initial velocity
        """
        self.m = m
        self.b = b
        self.dt = dt
        self.dyn_sigma = dyn_sigma
        self.max_init_vel = max_init_vel
        self._transition_matrix = self.transition_matrix
        super().__init__(n_balls, LinearBalls.STATE_DIM, img_size, episode_length, train_episodes, test_episodes, seed)

    def _initialize_states(self, n_balls):
        """ see super"""
        init_state = self.random.rand(n_balls, LinearBalls.STATE_DIM) * 2 - 1
        init_state[:, 2:] *= self.max_init_vel
        return init_state

    def _transition_function(self, state):
        """see super"""
        # state contains the states as row vectors (stacked to a matrix), so should the new states,
        # hence we can compute s_{t+1}^T = s_t^T A^T instead of s_{t+1} = A s_t
        n_balls = len(state)
        next_state = np.matmul(state, np.transpose(self._transition_matrix))
        next_state[:, 2:] += self.dyn_sigma * self.random.randn(n_balls, 2)
        return next_state

    def _get_task_space_states(self, states):
        """see super"""
        # for the linear balls the state space is the task space
        return states

    def has_task_space_velocity(self):
        """see super"""
        return True

    @property
    def transition_matrix(self):
        """Transition matrix of the linear ball tracking - copied from original implementation by T. Harnooja"""
        return np.array([[1,                 0,                 self.dt, 0      ],
                         [0,                 1,                 0,       self.dt],
                         [-self.dt / self.m, 0,                 self.b,  0      ],
                         [0,                 -self.dt / self.m, 0,       self.b ]])

    @staticmethod
    def observationMatrix():
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    @property
    def initial_state(self):
        return np.concatenate([self._states[:, 0, :], np.ones([self.number_of_episodes, 4])], axis=-1)

if __name__ == "__main__":
    data = LinearBalls(n_balls=1,
                       img_size=[64, 64, 3],
                       episode_length=100,
                       dyn_sigma=0.001,
                       train_episodes=3)

    print(np.max(data.visibility))

    pos = np.clip((data.positions * 32).astype(np.int) + 32, a_min=0, a_max=63)

    img_with_path = data.observations[0]
    img_with_path[:, pos[0, :, 1], pos[0, :, 0], 0] = 0 #np.arange(start=55, stop=255, step=2, dtype=np.uint8)
    img_with_path[:, pos[0, :, 1], pos[0, :, 0], 1] = np.arange(start=55, stop=255, step=2, dtype=np.uint8)
    img_with_path[:, pos[0, :, 1], pos[0, :, 0], 2] = np.arange(start=255, stop=55, step=-2, dtype=np.uint8)

    import matplotlib.pyplot as plt
    plt.imshow(img_with_path[0], interpolation="none")
    plt.show()

    data.images_to_vid(img_with_path, "/home/philipp/Code/dummy.avi")