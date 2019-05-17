import numpy as np
from PIL import Image
from PIL import ImageDraw
import data.ImgNoiseGeneration as noise_gen


class Pendulum:

    MAX_VELO_KEY = 'max_velo'
    MAX_TORQUE_KEY = 'max_torque'
    MASS_KEY = 'mass'
    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    FRICTION_KEY = 'friction'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'

    OBSERVATION_MODE_LINE = "line"
    OBSERVATION_MODE_BALL = "ball"

    def __init__(self,
                 img_size,
                 observation_mode,
                 generate_actions=False,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params = None,
                 seed=0):

        assert observation_mode == Pendulum.OBSERVATION_MODE_BALL or observation_mode == Pendulum.OBSERVATION_MODE_LINE
        # Global Parameters
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.observation_mode = observation_mode

        self.random = np.random.RandomState(seed)

        # image parameters
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55 if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE else 50
        self.plt_width = 8

        self.generate_actions = generate_actions

        # simulation parameters
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()
        self.max_velo = pendulum_params[Pendulum.MAX_VELO_KEY]
        self.max_torque = pendulum_params[Pendulum.MAX_TORQUE_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]
        self.mass = pendulum_params[Pendulum.MASS_KEY]
        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.inertia = self.mass * self.length**2 / 3
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]
        self.friction = pendulum_params[Pendulum.FRICTION_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std

        self.tranisition_covar_mat = np.diag(np.array([1e-8, self.transition_noise_std**2, 1e-8, 1e-8]))
        self.observation_covar_mat = np.diag([self.observation_noise_std**2, self.observation_noise_std**2])


    def sample_data_set(self, num_episodes, episode_length, full_targets, seed=None):
        if seed is not None:
            self.random.seed(seed)
        states = np.zeros((num_episodes, episode_length, self.state_dim))
        actions = self._sample_action((num_episodes, episode_length, self.action_dim))
        states[:, 0, :] = self._sample_init_state(num_episodes)

        for i in range(1, episode_length):
            states[:, i, :] = self._get_next_states(states[:, i - 1, :], actions[:, i - 1, :])
        states -= np.pi

        if self.observation_noise_std > 0.0:
            observation_noise = self.random.normal(loc=0.0,
                                                   scale=self.observation_noise_std,
                                                   size=states.shape)
        else:
            observation_noise = np.zeros(states.shape)

        targets = self.pendulum_kinematic(states)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            noisy_states = states + observation_noise
            noisy_targets = self.pendulum_kinematic(noisy_states)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            noisy_targets = targets + observation_noise
        imgs = self._generate_images(noisy_targets[..., :2])

        return imgs, targets[..., :(4 if full_targets else 2)], states, noisy_targets[..., :(4 if full_targets else 2)]

    @staticmethod
    def pendulum_default_params():
        return {
            Pendulum.MAX_VELO_KEY: 8,
            Pendulum.MAX_TORQUE_KEY: 10,
            Pendulum.MASS_KEY: 1,
            Pendulum.LENGTH_KEY: 1,
            Pendulum.GRAVITY_KEY: 9.81,
            Pendulum.FRICTION_KEY: 0,

            Pendulum.DT_KEY: 0.05,
            Pendulum.SIM_DT_KEY: 1e-4}

    def _sample_action(self, shape):
        if self.generate_actions:
            return self.random.uniform(-self.max_torque, self.max_torque, shape)
        else:
            return np.zeros(shape=shape)

    def _transition_function(self, states, actions):
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = states[..., 1:2] + self.sim_dt * (c * np.sin(states[..., 0:1])
                                                     + actions / self.inertia
                                                     - states[..., 1:2] * self.friction)
            states = np.concatenate((states[..., 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states

    def _get_next_states(self, states, actions):
        actions = np.maximum(-self.max_torque, np.minimum(actions, self.max_torque))

        states = self._transition_function(states, actions)
        if self.transition_noise_std > 0.0:
            states[:, 1] += self.random.normal(loc=0.0,
                                               scale=self.transition_noise_std,
                                               size=[len(states)])

        states[:, 0] = ((states[:, 0]) % (2 * np.pi))
        return states

    def get_ukf_smothing(self, obs):
        batch_size, seq_length = obs.shape[:2]
        succ = np.zeros(batch_size, dtype=np.bool)
        means = np.zeros([batch_size, seq_length, 4])
        covars = np.zeros([batch_size, seq_length, 4, 4])
        fail_ct = 0
        for i in range(batch_size):
            if i % 10 == 0:
                print(i)
            try:
                means[i], covars[i] = self.ukf.filter(obs[i])
                succ[i] = True
            except:
                fail_ct +=1
        print(fail_ct / batch_size, "failed")

        return means[succ], covars[succ], succ

    def _sample_init_state(self, nr_epochs):
        return np.concatenate((self.random.uniform(0, 2 * np.pi, (nr_epochs, 1)), np.zeros((nr_epochs, 1))), 1)

    def add_observation_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        return noise_gen.add_img_noise(imgs, first_n_clean, self.random, r, t_ll, t_lu, t_ul, t_uu)

    def _get_task_space_pos(self, joint_states):
        task_space_pos = np.zeros(list(joint_states.shape[:-1]) + [2])
        task_space_pos[..., 0] = np.sin(joint_states[..., 0]) * self.length
        task_space_pos[..., 1] = np.cos(joint_states[..., 0]) * self.length
        return task_space_pos

    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos.shape)[:-1] + [self.img_size, self.img_size], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]):
            for idx in range(ts_pos.shape[1]):
                imgs[seq_idx, idx] = self._generate_single_image(ts_pos[seq_idx, idx])

        return imgs

    def _generate_single_image(self, pos):
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        img = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            x_l = x1 - self.plt_width
            x_u = x1 + self.plt_width
            y_l = y1 - self.plt_width
            y_u = y1 + self.plt_width
            draw.ellipse((x_l, y_l, x_u, y_u), fill=1.0)

        img = img.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(img)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array

    def _kf_transition_function(self, state, noise):
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = state[1] + self.sim_dt * (c * np.sin(state[0]) - state[1] * self.friction)
            state = np.array([state[0] + self.sim_dt * velNew, velNew])
        state[0] = state[0] % (2 * np.pi)
        state[1] = state[1] + noise[1]
        return state

    def pendulum_kinematic_single(self, js):
        theta, theat_dot = js
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theat_dot *  y
        y_dot = theat_dot * -x
        return np.array([x, y, x_dot, y_dot]) * self.length

    def pendulum_kinematic(self, js_batch):
        theta = js_batch[..., :1]
        theta_dot = js_batch[..., 1:]
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theta_dot * y
        y_dot = theta_dot * -x
        return np.concatenate([x, y, x_dot, y_dot], axis=-1)

    def inverse_pendulum_kinematics(self, ts_batch):
        x = ts_batch[..., :1]
        y = ts_batch[..., 1:2]
        x_dot = ts_batch[..., 2:3]
        y_dot = ts_batch[..., 3:]
        val = x / y
        theta = np.arctan2(x, y)
        theta_dot_outer = 1 / (1 + val**2)
        theta_dot_inner = (x_dot * y - y_dot * x) / y**2
        return np.concatenate([theta, theta_dot_outer * theta_dot_inner], axis=-1)


if __name__ == '__main__':

    img_size = 24

    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0.1

    data = Pendulum(img_size=img_size,
                    observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1,
                    observation_noise_std=1e-5,
                    pendulum_params=pend_params,
                    seed=0)

    samples, _, _, _ = data.sample_data_set(500, 150, full_targets=False, seed=1)
    noisy_samples, factors = data.add_observation_noise(samples, 0)

    np.savez("pend_test", images=samples)


    print("test")


