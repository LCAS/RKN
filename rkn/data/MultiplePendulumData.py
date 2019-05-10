from data.PendulumData import Pendulum
import numpy as np
from PIL import Image
from PIL import ImageDraw
from data.ImgNoiseGeneration import add_img_noise4

class MultiplePendulums(Pendulum):

    def __init__(self,
                 img_size,
                 num_pends,
                 observation_mode,
                 generate_actions=False,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params = None,
                 seed=0):
        super().__init__(img_size=img_size, observation_mode=observation_mode, generate_actions=generate_actions,
                         transition_noise_std=transition_noise_std, observation_noise_std=observation_noise_std,
                         pendulum_params=pendulum_params, seed=seed)
        self._num_pends = num_pends
        self.state_dim = 2 * num_pends
        self.colors = [(1.0, 0.0, 0.0),
                       (-2.0, 1.0, 0),
                       (-2.0, -2.0, 1,0)]

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
        imgs = self._generate_images(noisy_targets)

        targets = targets if full_targets else targets[..., [0,1, 4,5, 8,9]]
        return imgs, targets, states, noisy_targets[..., :(4 if full_targets else 2)]

    def _sample_init_state(self, nr_epochs):
        init_states = []
        for _ in range(self._num_pends):
            init_states.append(super()._sample_init_state(nr_epochs))
        return np.column_stack(init_states)

    def _get_next_states(self, states, actions):
        next_states = []
        for i in range(self._num_pends):
            next_states.append(super()._get_next_states(states[:, 2 * i : 2 * (i+1)], actions=actions))
        return np.concatenate(next_states, -1)

    def pendulum_kinematic(self, js_batch):
        pos = []
        for i in range(self._num_pends):
            pos.append(super().pendulum_kinematic(js_batch[..., 2 * i : 2 * (i+1)]))
        return np.concatenate(pos, -1)

    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos.shape)[:-1] + [self.img_size, self.img_size, 3], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]):
            for idx in range(ts_pos.shape[1]):
                imgs[seq_idx, idx] = self._generate_single_image(ts_pos[seq_idx, idx])
        return imgs

    def _generate_single_image(self, pos):

        imgs = []
        for i in range(self._num_pends):
            img = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
            draw = ImageDraw.Draw(img)
            x1 = pos[4 * i + 0] * (self.plt_length / self.length) + self.x0
            y1 = pos[4 * i + 1] * (self.plt_length / self.length) + self.y0
            if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
                draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
            elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
                x_l = x1 - self.plt_width
                x_u = x1 + self.plt_width
                y_l = y1 - self.plt_width
                y_u = y1 + self.plt_width
                draw.ellipse((x_l, y_l, x_u, y_u), fill=self.colors[i])
            img = img.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
            imgs.append(np.asanyarray(img))

        img = np.concatenate([np.expand_dims(img, -1) for img in imgs], -1)

        img[np.logical_or(img[..., 1] > 0.1, img[..., 2] > 0.1), 0] = 0.0
        img[img[..., 2] > 0.1, 1] = 0.0

        img = 255.0 * np.clip(img, 0, 1)
        return  img

    def add_observation_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        return add_img_noise4(imgs, first_n_clean, self.random, r, t_ll, t_lu, t_ul, t_uu)


if __name__ == '__main__':
    import matplotlib
   # matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    def nll(prediction, target, variance):
        element_wise_ll = 0.5 * (((prediction - target) ** 2) / variance + np.log(variance) + np.log(2 * np.pi))
        sample_wise_ll = np.sum(element_wise_ll, axis=-1)
        return np.mean(sample_wise_ll)

    def full_nll(prediction, target, variance):
        dim = prediction.shape[-1]
        constant_term = dim * np.log(2 * np.pi)
        reg_term = np.log(np.linalg.det(variance))
        diff = prediction - target
        loss_term = np.tensordot(diff, np.linalg.solve(variance, diff), axes=[[2], [2]])
        return np.mean(0.5 * (constant_term + reg_term + loss_term))

    def rmse(prediction, target):
        return np.sqrt(np.mean((prediction - target) ** 2))

    img_size = 24

    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0.1

    data = MultiplePendulums(img_size=img_size,
                             num_pends=3,
                    observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1,
                    observation_noise_std=0.1,
                    pendulum_params=pend_params,
                    seed=0)
    samples, _, _, _ = data.sample_data_set(10, 150, full_targets=False)
    noisy_samples, q = data.add_observation_noise(samples, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)

    from plotting.PendulumPlotting import PendulumPlotter

    p = PendulumPlotter("dummy", "dummy", plot_n=10)
    p.plot_observation_sequences(noisy_samples, noisy_samples, noisy_samples)


   # fig = plt.figure()
 #   img_input = plt.imshow(noisy_samples[0, 0], interpolation="None")
#
   # def update(i):
   #     print(i)
   #     seq_idx = int(np.floor(i/150))
   #     img_idx = i % 150
   #     img_input.set_data(noisy_samples[seq_idx, img_idx])

    #ani = anim.FuncAnimation(fig, update, 1500)
    #writer = anim.writers['ffmpeg'](fps=10)
    #ani.save("test.mp4", writer=writer)


   # plt.figure()
    for i in range(samples.shape[0]):
        print("New Episode")
        for j in range(samples.shape[1]):
            plt.imshow(noisy_samples[i, j], interpolation="none")
            plt.pause(.01)
            plt.show()


    print("test")