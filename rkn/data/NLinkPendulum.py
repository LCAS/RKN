import _DoubleLinkForwardModel as DoubleLink
import _QuadLinkForwardModel as QuadLink
import numpy as np
#import data.ImgNoiseGeneration as noise_gen
import data.ImgNoiseGeneration_harit_v4 as noise_gen
from PIL import Image
from PIL import ImageDraw

class NLinkPendulum:

    DL = "double"
    QL = "quad"

    def __init__(self,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 pendulum,
                 generate_img_noise,
                 keep_clean_imgs=False,
                 dt=.05,
                 sim_dt=1e-4,
                 friction=None,
                 first_n_clean=5,
                 seed=0):
        assert pendulum == NLinkPendulum.DL or pendulum == NLinkPendulum.QL, "Invalid Pendulum"
        self.links = 2 if pendulum == NLinkPendulum.DL else 4
        self._transition_function = self._transition_function_double if pendulum == NLinkPendulum.DL else self._transition_function_quad
        self.dt = dt
        self.sim_dt = sim_dt
        self.masses =  np.ones(self.links)
        self.lengths = np.ones(self.links)
        self.inertias = self.masses * (self.lengths ** 2) / 3
        self.g = 9.81
        self.friction = friction if friction is not None else np.zeros(self.links)
        self.random = np.random.RandomState(seed)
        self._img_size_internal = 128
        self._vis_line_width = 6
        self._first_n_clean = first_n_clean
        self.generate_img_noise = generate_img_noise
        self.keep_clean_imgs = keep_clean_imgs if generate_img_noise else False
        self.img_size = 48

        self.train_states, self.train_angles = self._sample_sequences(train_episodes, episode_length)
        self.test_states, self.test_angles = self._sample_sequences(test_episodes, episode_length)
        if self.keep_clean_imgs and self.generate_img_noise:
            self.train_images, self.train_factors, self._train_images_clean = self._render(self.train_states)
            self.test_images, self.test_factors, self._test_images_clean = self._render(self.test_states)
        else:
            self.train_images, self.train_factors = self._render(self.train_states)
            self.test_images, self.test_factors = self._render(self.test_states)

    @property
    def train_images_clean(self):
        if self.generate_img_noise:
            if self.keep_clean_imgs:
                return self._train_images_clean
            else:
                raise AssertionError("If noise is generated keep_clean_images needs to be true if they are required")
        else:
            return self.train_images

    @property
    def test_images_clean(self):
        if self.generate_img_noise:
            if self.keep_clean_imgs:
                return self._test_images_clean
            else:
                raise AssertionError("If noise is generated keep_clean_images needs to be true if they are required")
        else:
            return self.test_images



    def _sample_sequences(self, num_seqs, seq_length):
        self.states = np.zeros([num_seqs, seq_length, self.links * 2])
        pos = [self.random.uniform(-np.pi, np.pi, (num_seqs, 1)) for _ in range(self.links)]
        #vel = [self.random.uniform(-1, 1, (num_seqs, 1)) for _ in range(self.links)]
        vel = [np.zeros([num_seqs, 1]) for _ in range(self.links)]
        self.states[:, 0] = np.concatenate([val for pair in zip(pos, vel) for val in pair], -1)
        for i in range(seq_length-1):
            self.states[:, i + 1] = self._transition_function(self.states[:, i])
        return self.states, self.states[:, :, ::2]

    def _render(self, states):
        images = np.zeros([states.shape[0], states.shape[1], self.img_size, self.img_size, 1], dtype=np.uint8)
        for i in range(states.shape[0]):
            for j in range(states.shape[1]):
                img = np.expand_dims(self._render_single_image(states[i, j]), -1)
                images[i, j] = img
        if self.generate_img_noise:
            imgs_noisy, factors = noise_gen.add_img_noise(images, self._first_n_clean, self.random,
                                                          r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
            imgs_noisy = np.expand_dims(imgs_noisy, -1)
        else:
            imgs_noisy = images
        if self.generate_img_noise and self.keep_clean_imgs:
            return imgs_noisy, factors, images
        else:
            del images
            return imgs_noisy, np.ones([imgs_noisy.shape[0], imgs_noisy.shape[1], 1, 1, 1])

    def _render_single_image(self, state):
        img = Image.new('F', (self._img_size_internal, self._img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        start = (self._img_size_internal / 2, self._img_size_internal / 2)
        vis_link_length = self._img_size_internal / (2 * self.links)
        for i in range(self.links):
            if i == 0 :
                end = (start[0] + np.sin(state[2 * i]) * vis_link_length, start[1] + np.cos(state[2 * i]) * vis_link_length)
            else:
                end = (start[0] - np.sin(state[2 * i]) * vis_link_length, start[1] - np.cos(state[2 * i]) * vis_link_length)

            draw.line([start, end], fill=1.0, width=self._vis_line_width)
            start = end

        img = img.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.array(img)
        img_as_array = img_as_array.clip(0.0, 1.0)
        return 255.0 * np.array(img_as_array)

    def _transition_function_double(self, state):
        """see super"""
        actions = np.zeros((state.shape[0], 2))
        result = np.zeros((state.shape[0], 6))
        DoubleLink.simulate(state, actions, self.dt, self.masses, self.lengths, self.inertias,
                            self.g, self.friction, self.sim_dt, 0, np.zeros(4), np.zeros(4), result)
        return result[:, :4]

    def _transition_function_quad(self, state):
        actions = np.zeros((state.shape[0], 4))
        result = np.zeros((state.shape[0], 12))
        QuadLink.simulate(state, actions, self.dt, self.masses, self.lengths, self.inertias,
                         self.g, self.friction, self.sim_dt, result)
        return result[:, :8]


    def images_to_vid(self, images, filename):
        """
        Generates video out of sequence of images
        :param images: sequence of images
        :param filename: filname to save video under (including path)
        :return:
        """
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = plt.gca()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        img_dummy = axes.imshow(images[0], interpolation='none', cmap='gray')
        ani = anim.FuncAnimation(fig, lambda n: img_dummy.set_data(images[n]), len(images))
        writer = anim.writers['ffmpeg'](fps=20)
        ani.save(filename, writer=writer, dpi=100)
        plt.close()

    def to_sc_representation(self, angles):
        l = []
        for i in range(self.links):
            l += [np.expand_dims(np.sin(angles[..., i]), -1), np.expand_dims(np.cos(angles[..., i]), -1)]
        return np.concatenate(l, -1)

    def from_sc_representation(self, sc_represented_angles):
        l = []
        for i in range(self.links):
            l.append(np.expand_dims(np.arctan2(sc_represented_angles[..., 2*i], sc_represented_angles[..., 2*i+1]), -1))
        return np.concatenate(l, -1)


if __name__ == "__main__":
    data = NLinkPendulum(episode_length=150,
                         train_episodes=50,
                         test_episodes=50,
                         first_n_clean=5,
                         pendulum=NLinkPendulum.QL,
                         generate_img_noise=False,
                         keep_clean_imgs=False,
                         friction=0.1 * np.ones(4),
                         seed=0,
                         dt=0.05)

    np.savez("ql", images=data.train_images)
    np.savez("ql_test", images=data.test_images)

    #data.images_to_vid(data.train_images[3, :, -1:0:-1, :], "/home/philipp/RKN/vid.avi")

    #for friction in [
    # 0.0, 0.1, 0.2, 0.3]:
    #    for dt in [0.05, 0.025, 0.01]:
    #        d = NLinkPendulum(episode_length=150, train_episodes=100, test_episodes=10, pendulum=NLinkPendulum.QL,
    #                          generate_img_noise=False, friction=friction * np.ones(4), first_n_clean=5, dt=dt)

    #        res = d.from_sc_representation(d.to_sc_representation(d.train_angles))
    #        transformed = (d.train_angles + np.pi) % (2 * np.pi) - np.pi
    #        assert np.all(np.isclose(res, transformed)), "worng representation transform"
    #        imgs, _ = d.add_observation_noise(d.train_images, 0)
    #        name = "QL_" + str(friction) + "_" +str(dt)
    #        d.images_to_vid(d.train_images[5, :, -1:0:-1, :, 0], "/home/philipp/RKN/" + name + ".avi")

    #print("Completely visible: ", np.count_nonzero(d.train_factors == 1) / d.train_factors.size)
    #print("Completely invisible: ", np.count_nonzero(d.train_factors == 0) / d.train_factors.size)

    #plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.hist(np.ravel(d.train_factors), normed=True, bins=20)
    #plt.title("Train")
    #plt.subplot(1, 2, 2)
    #plt.hist(np.ravel(d.test_factors), normed=True, bins=20)
    #plt.show()
   # t = d.to_sc_representation(d.train_angles)
  #  plt.figure()
  #  plt.plot(t[0, :, 0])
  #  plt.plot(t[0, :, 1])
  #  plt.plot(t[0, :, 2])
  #  plt.plot(t[0, :, 3])
  #  plt.plot(t[0, :, 4])
  #  plt.plot(t[0, :, 5])
  #  plt.plot(t[0, :, 6])
   # plt.plot(t[0, :, 7])



#    plt.show()

