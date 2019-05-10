import abc
import numpy as np
from PIL import Image
from PIL import ImageDraw

class AbstractBalls(abc.ABC):
    """Abstract Superclass to generate data for Ball tracking task, Code based on implementation provided by the
    courtesy of T.Haarnoja"""
    TRACK_BALL_COLOR = np.array([255, 0, 0], dtype=np.uint8)

    def __init__(self,
                 n_balls,
                 state_dim,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 seed):
        """
        Creates new dataset
        :param n_balls: number of balls in the image (if this is a negative number -n, the number of balls is sampled
        between 0 and n. A new number is sampled for each sequence
        :param state_dim: dimensionality of the state (given by subclasses)
        :param img_size: size of the images to generate
        :param episode_length: length of the sequences that will be generated
        :param train_episodes: number of sequences that will be generated
        :param seed: seed for the random number generator
        """

        self.random = np.random.RandomState(seed)
        self.n_balls = n_balls
        self.state_dim = state_dim
        self.img_size = img_size
        #generate one more - first is initial
        self.episode_length = episode_length + 1

        self._train_images, self._train_states = self._simulate_data(train_episodes)
        self._train_visibility = self.compute_visibility(self._train_images)

        self._test_images, self._test_states = self._simulate_data(test_episodes)
        self._test_visibility = self.compute_visibility(self._test_images)

    def _simulate_data(self, number_of_episodes):
        """
        Simulates the dataset
        :return: images (observations) and task_space_states (positions (+ velocity if 'has_task_space_velocity')) for the
        ball to track
        """


        images = np.zeros([number_of_episodes, self.episode_length] + self.img_size, dtype=np.uint8)
        ts_dim = 4 if self.has_task_space_velocity else 2
        task_space_states = np.zeros([number_of_episodes, self.episode_length, ts_dim])

        for i in range(number_of_episodes):
            # +1 since randint samples from an interval excluding the high value
            n_balls = self.n_balls if self.n_balls > 0 else self.random.randint(low=1, high=-self.n_balls + 1)
            states = np.zeros([self.episode_length, n_balls, self.state_dim])
            states[0, :, :] = self._initialize_states(n_balls)
            for j in range(1, self.episode_length):
                states[j, :, :] = self._transition_function(states[j - 1, :, :])
            all_task_space_states = self._get_task_space_states(states)
            images[i] = self._render(all_task_space_states)
            task_space_states[i] = all_task_space_states[:, 0]
        return images, task_space_states


    def _render(self, task_space_states):
        """
        Creates images out of positions
        :param task_space_states: Batches of sequences of the task space states of all balls as an
               [episode_length x number_of_balls x task space dim]
        :return: sequence of images
        """

        images = np.zeros([self.episode_length] + self.img_size, dtype=np.uint8)
        n_balls = task_space_states.shape[1]
        radii = self.random.randint(low=5, high=10, size=n_balls)
        radii[0] = 5
        # Those magic numbers where chosen to match the original ball data implementation by T. Haarnoja
        radii = np.floor(radii * self.img_size[0] / 128).astype(np.int)

        colors = self.random.randint(low=0, high=255, size=[n_balls, 3])
        colors[0] = AbstractBalls.TRACK_BALL_COLOR
        for i in range(self.episode_length):
            img = Image.new('RGB', self.img_size[:2])
            draw = ImageDraw.Draw(img)
            for j in range(n_balls):
                x, y = task_space_states[i, j, :2]
                x = np.int64((x + 1) * self.img_size[0] / 2)
                y = np.int64((y + 1) * self.img_size[1] / 2)
                r = radii[j]
                draw.ellipse((x-r, y-r, x+r, y+r), fill=tuple(colors[j]), outline=tuple(colors[j]))
            images[i] = np.array(img)
        return images

    def images_to_vid(self, images, filename):
        """
        Generates video out of sequence of images
        :param images: sequence of images
        :param filename: filname to save video under (including path)
        :return:
        """
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        assert len(images) == self.episode_length - 1, "Length of given sequence does not match sequence length, something wrong"
        fig = plt.figure()
        axes = plt.gca()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        img_dummy = axes.imshow(images[0], interpolation='none')
        ani = anim.FuncAnimation(fig, lambda n: img_dummy.set_data(images[n]), len(images))
        writer = anim.writers['ffmpeg'](fps=10)
        ani.save(filename, writer=writer, dpi=100)
        plt.close()

    def compute_visibility(self, observations):
        """Counts how many pixels of the ball are visible in each image
        :param observations: batch of sequences of images"""
        color_reshape = np.reshape(AbstractBalls.TRACK_BALL_COLOR, [1, 1, 1, 1, 3])

        # this subtraction also works in uint8 (its not a reasonable value but it is zero if and only if it should be)
        observations = observations - color_reshape        # subtract ball color from images - count how many are zero

        # count how many are zeros there are for each image
        pixel_is_color = np.all(observations == 0, axis=-1)
        values = np.sum(pixel_is_color, (2, 3))

        return values

    @property
    def train_positions(self):
        """
        The true positions of the ball from t=1
        :return:
        """
        return self._train_states[:, 1:, :2]

    @property
    def test_positions(self):
        return self._test_states[:, 1:, :2]

    @property
    def train_observations(self):
        """
        The observations of the ball from t=1
        :return:
        """
#        print(np.max(self._images), np.average(self._images))
        return self._train_images[:, 1:]

    @property
    def test_observations(self):
        return self._test_images[:, 1:]

    @property
    def train_visibility(self):
        return self._train_visibility[:, 1:]

    @property
    def test_visibility(self):
        return self._test_visibility[:, 1:]

    @property
    def initial_state(self):
        """The initial latent state at t=0 (may not be given)"""
        raise NotImplementedError("Initial latent state unknown and can not be given!")


    @abc.abstractmethod
    def _initialize_states(self, n_balls):
        """Sample initial states for all balls"""
        raise NotImplementedError("State Initialization not implemented")

    @abc.abstractmethod
    def _get_task_space_states(self, states):
        """
        Map states to task space states, needs to be capable to handle sequences of batches (of n balls)
        :param states: states in not task space (e.g. joint space for robotic system or n-link pendulum
        :return: states in task space (location of center of the ball in the image, if 'has_task_space_velocity'
        this also needs to return the velocity of the ball (in the image)
        """
        raise NotImplementedError("Task Space Mapping not implemented")

    @abc.abstractmethod
    def _transition_function(self, state):
        """
        Maps from current state to next state, needs to be capable of handling batches (of n balls)
        :param state: current state
        :return: next state
        """
        raise NotImplementedError("Transition Function not implemented")

    @property
    def has_task_space_velocity(self):
        raise NotImplementedError("Has task space velocity not implemented ")