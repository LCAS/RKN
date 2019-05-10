import numpy as np
from data.AbstractBalls import AbstractBalls
import _QuadLinkForwardModel as QuadLink

class QuadLinkBalls(AbstractBalls):
    STATE_DIM = 8

    def __init__(self,
                 n_balls,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 dt=0.05,
                 sim_dt=1e-4,
                 masses=None,
                 lenghts=None,
                 g=9.81,
                 friction=None,
                 seed=None,
                 scale_factor = None):
        self.dt = dt
        self.sim_dt = sim_dt
        self.masses = masses if masses is not None else np.ones(4)
        self.lengths = lenghts if lenghts is not None else np.ones(4)
        self.inertias = self.masses * (self.lengths ** 2) / 3
        self.g = g
        self.friction = friction if friction is not None else np.zeros(4)
        self.scale_factor = scale_factor
       # print("scale_factor: ", scale_factor)

        super().__init__(n_balls, QuadLinkBalls.STATE_DIM, img_size, episode_length, train_episodes, test_episodes, seed)

    def _initialize_states(self, n_balls):
        """ see super """
        p1, p2, p3, p4 = [self.random.uniform(-np.pi, np.pi, (n_balls, 1)) for _ in range(4)]
        v1, v2, v3, v4 = [self.random.uniform(-1, 1, (n_balls, 1)) for _ in range(4)]
        return np.concatenate([p1, v1, p2, v2, p3, v3, p4, v4], axis=-1)


    def _transition_function(self, state):
        """ see super """
        actions = np.zeros((state.shape[0], 4))
        result = np.zeros((state.shape[0], 12))
        QuadLink.simulate(state, actions, self.dt, self.masses, self.lengths, self.inertias,
                          self.g, self.friction, self.sim_dt, result)
        return result[:, :8]

    def _get_task_space_states(self, states):
        """ see super """
        n_balls = states.shape[1]
        positions = np.zeros((self.episode_length, n_balls, 2))
        for i in range(4):
            positions[:, :, 0] += np.sin(states[:, :, 2 * i]) * self.lengths[i]
            positions[:, :, 1] += np.cos(states[:, :, 2 * i]) * self.lengths[i]
        if self.scale_factor is not None:
            return positions * self.scale_factor
        # map to interval [-1, 1], i.e. the plotted region
        else:
            return positions / np.sum(self.lengths)

    @property
    def has_task_space_velocity(self):
        """see super"""
        return False

    @property
    def initial_state(self):
        return None

if __name__ == "__main__":
    data = QuadLinkBalls(n_balls=10,
                         img_size=[64, 64, 3],
                         episode_length=100,
                         train_episodes=3)


    pos = (data.positions * 32).astype(np.int) + 32

    img_with_path = data.observations[0]
    img_with_path[:, pos[0, :, 1], pos[0, :, 0], 1] =  np.arange(start=55, stop=255, step=2, dtype=np.uint8)
    img_with_path[:, pos[0, :, 1], pos[0, :, 0], 2] = np.arange(start=255, stop=55, step=-2, dtype=np.uint8)


    data.images_to_vid(img_with_path, "/home/philipp/Code/dummy2.avi")

