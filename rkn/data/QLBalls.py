from __future__ import print_function
import numpy as np
import cv2
import _QuadLinkForwardModel as QuadLink
floatX = np.float32

'''Modified Balls Code '''
class QLBalls:
    def __init__(self,
                 n_balls,  # total number of balls, max number if negative
                 dyn_sig,  # dynamics params
                 w, h,  # width and heights of the frame
                 seed=0,  # random seed,
                 data_size=-1,  # reset seed after data_size sequences
                 seq_len=100,  # sequence length, used only if cache=True
                 cache=False,  # cache sequences
                 shuffle=False,
                 **kwargs):  # shuffle data for pretraining (needs cache=1)

        print("BALL DATA QUAD LINK, nballs", n_balls)
        self.maxVelo = kwargs['maxVelo'] if 'maxVelo' in kwargs else 8
        self.maxTorque = kwargs['maxTorque'] if 'maxTorque' in kwargs else 10
        #self.dt = kwargs['dt'] if 'dt' in kwargs else 1e-2
        self.masses = kwargs['mass'] if 'mass' in kwargs else np.ones(4)
        self.lengths = kwargs['length'] if 'length' in kwargs else np.ones(4)
        self.inertias = self.masses * self.lengths**2 / 3
        self.g = kwargs['g'] if 'g' in kwargs else 9.81
        self.friction = kwargs['friction'] if 'friction' in kwargs else np.zeros(4)
        self.sim_dt = kwargs['sim_dt'] if 'sim_dt' in kwargs else 1e-4


        self.Dx = 2
        self.w = w
        self.h = h
       # self.max_vel = 0.1
        self.seed = seed
        self.rand = np.random.RandomState(seed)
        if n_balls > 0:
            self.n_balls = n_balls
            self.max_balls = []
        else:
            self.n_balls = []
            self.max_balls = - n_balls
        self.dt = 0.05
        self.x0cov_const = 0.001 * np.eye((self.Dx))

        self.dyn_sigma = dyn_sig  # dynamics std

        self.o_dim = (3, self.h, self.w)
        self.im_dim = (self.h, self.w, 3)
        self.x_dim = (4,)
        self.y_dim = (2,)

        self.line_true = []
        self.line_pred = []
        self.o = []
        self.data_size = data_size
        self.sampled = False
        self.seq_counter = 0

        self.cache = cache
        self.shuffle = shuffle
        self.seq_len = seq_len
        if self.cache:
            self.collect(self.seq_len, data_size)
        if self.shuffle and self.cache:
            self.o = self.o.reshape((-1,) + self.o_dim)
            self.y = self.y.reshape((-1,) + self.y_dim)
            rand_ind = self.rand.permutation(self.y.shape[0])
            self.o = self.o[rand_ind].reshape((data_size, self.seq_len,) +
                                              self.o_dim)
            self.y = self.y[rand_ind].reshape((data_size, self.seq_len,) +
                                              self.y_dim)

    def init_state(self, steps):
        # if max_balls is set, then we choose the number of
        # balls at random
        if self.max_balls:
            self.n_balls = self.rand.randint(self.max_balls) + 1
        self.board = np.zeros((steps,) + self.im_dim, dtype=np.uint8)
        self.j_state = np.zeros([self.n_balls, steps, 8])
        self.j_state[:, 0, :] = self._sample_init_state(self.n_balls)
        self.state = np.zeros([self.n_balls, steps, 2])
        self.state[:, 0, :2] = self._generate_observations(self.j_state[:, 0, :])
        self.colors = np.zeros((self.n_balls, 3))

        self.r = np.uint8(self.rand.rand(self.n_balls) * 2.5 + 2.5)
        self.r[0] = 3  # always set the first ball the same size

        # the first ball is red and the other are or random color
        self.colors[0] = [255, 0, 0]
        if self.n_balls > 1:
            self.colors[1:] = self.rand.randint(255, size=((self.n_balls - 1), 3))

    def collect(self, seq_len, n_seqs):
        """Generate n_seq sequences of length seq_len and store them in o and y"""
        self.o = np.zeros((n_seqs, seq_len) + self.o_dim, dtype=np.uint8)
        self.y = np.zeros((n_seqs, seq_len) + self.y_dim, dtype=np.float32)
        self.x0 = np.zeros((n_seqs, self.Dx)).astype(np.float32)
        self.x0cov = np.zeros((n_seqs, self.Dx, self.Dx)).astype(np.float32)
        for i in range(n_seqs):
            self.simulate(seq_len)
            self.o[i] = self.board.transpose((0, 3, 1, 2))
            self.y[i] = self.state[0, :, :2]  # pick the first ball
            self.x0[i] = self.state[0, 0]
            self.x0cov[i] = self.x0cov_const

        self.visible = self.compute_visibility(self.o, np.array([255, 0, 0], dtype=np.uint8))

    def sample(self, seq_len, n_seqs=1):
        if not self.cache:
            self.collect(seq_len, n_seqs)
            seqs = np.array(range(n_seqs))
        else:
            seqs = (np.array(range(n_seqs)) + self.seq_counter) % self.data_size
            self.seq_counter += n_seqs

        o, y = self.o[seqs] / 255.0, self.y[seqs]
        x0, x0cov = self.x0[seqs], self.x0cov[seqs]

        return o.astype(np.float32), y.astype(np.float32), \
               (x0.astype(np.float32), x0cov.astype(np.float32))

    def draw(self, t):
        for i in range(self.n_balls):
            x, y = self.state[i, t, :2]
            x = np.int64((x + 1) * self.w / 2)
            y = np.int64((y + 1) * self.h / 2)
            cv2.circle(self.board[t], (y, x), self.r[i], self.colors[i], -1)

    def animate(self, seq_num=0, target=[], radius=4, filename=None, draw=True):

        if filename is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
           # fourcc = cv2.cv2.CV_FOURCC(*'HFYU')
            v_writer = cv2.VideoWriter(filename, fourcc, 10.0, (self.w, self.h))
            #out = cv2.VideoWriter(filename, fourcc, 10.0, (self.w, self.h))

        self.fig = plt.imshow(np.zeros(self.im_dim).astype(np.uint8))
        for t in range(self.seq_len):
            brd = self.o[seq_num, t].transpose((1, 2, 0)).copy()
            for i in range(len(target)):
                x = np.int64((target[i][t, 0] + 1) * self.w / 2)
                y = np.int64((target[i][t, 1] + 1) * self.h / 2)
                if isinstance(radius, int):
                    cv2.circle(brd, (y, x), radius, (255,) * 3, 2)
                else:
                    cv2.circle(brd, (y, x), radius[t], (255,) * 3, 2)
            if filename is not None:
                v_writer.write(brd[:, :, [2, 1, 0]])

            if draw:
                self.fig.set_data(brd)
                plt.draw()
                plt.pause(0.001)

        if filename is not None:
            v_writer.release()

    def montage(self, seq_num, start, frames, step=1):
        im = self.o[seq_num, start:start + frames * step:step]
        s = self.y[seq_num, start]
        x = np.int64((s[0] + 1) * self.w / 2)
        y = np.int64((s[1] + 1) * self.h / 2)
        frame1 = im[0].transpose((1, 2, 0)).copy()
        cv2.circle(frame1, (y, x), 15, (255,) * 3, 2)
        im[0] = frame1.transpose((2, 0, 1))

        dim = im.shape
        margin = 5
        zeros = 255 * np.ones((dim[0], dim[1], dim[2], margin)).astype(np.uint8)
        im = np.concatenate((im, zeros), axis=3)
        im = im.transpose((2, 0, 3, 1))
        im = im.reshape((self.h, -1, 3))
        im = im[:, :-margin, :]
        plt.imshow(im)
        plt.axis('off')
        plt.show()

    def simulate(self, steps=100):
        # reset random seed after data_size sequences
        if self.data_size > 0 and self.seq_counter % self.data_size == 0:
            self.rand.seed(self.seed)
        self.seq_counter += 1
        self.init_state(steps)
        self.draw(0)
        for t in range(steps - 1):
            self.j_state[:, t + 1, :] = self._transition_function(self.j_state[:, t, :])
            self.state[:, t + 1, :] = self._generate_observations(self.j_state[:, t + 1, :]) / 4

            #self.state[:, t + 1, :2] = self.state[:, t, :2] + self.dt * self.state[:, t, 2:]
            #self.state[:, t + 1, 2:] = self.b * self.state[:, t, 2:] \
            #                           - self.dt / self.m * (self.state[:, t, :2]) \
#                                       + self.dyn_sigma * self.rand.randn(*(self.n_balls, 2,))
            self.draw(t + 1)

    def _transition_function(self, states):
        actions = np.zeros((states.shape[0], 4))
        if len(np.shape(states)) == 1:
            states = np.expand_dims(states, 0)
            actions = np.expand_dims(actions, 0)
        num_samples = len(states)
        if self.dt - 1e-3 < 1.0 < self.dt + 1e-3:
            result = np.zeros((num_samples, 4))
        else:
            result = np.zeros((num_samples, 12))
        QuadLink.simulate(states, actions, self.dt, self.masses, self.lengths, self.inertias, self.g,
                          self.friction, self.sim_dt, result)
        return result[:, :8]

    def _generate_observations(self, states):
        positions = np.zeros((states.shape[0], 2))
        for i in range(4):
            positions[:, 0] += np.sin(states[:, 2 * i]) * self.lengths[i]
            positions[:, 1] += np.cos(states[:, 2 * i]) * self.lengths[i]
        return positions

    def _sample_init_state(self, nr_epochs):
        init_state = np.zeros((nr_epochs, 8))
        init_pos = np.random.uniform(-np.pi, np.pi, (nr_epochs, 4))
        init_vel = np.random.uniform(-1, 1, (nr_epochs, 4))
        for i in range(4):
            init_state[:, 2 * i] = init_pos[:, i]
            init_state[:, 2 * i + 1] = init_vel[:, i]
        return init_state


    def plot_comparison(self, y, yhat):
        if not self.line_true:
            self.line_true, = plt.plot(1, 1, label='ground truth')
            self.line_pred, = plt.plot(1, 1, label='prediction')
            plt.legend()

        self.line_true.set_xdata(y[:, 0])
        self.line_true.set_ydata(y[:, 1])
        self.line_pred.set_xdata(yhat[:, 0])
        self.line_pred.set_ydata(yhat[:, 1])
        margin = .1
        plt.axis([min(yhat[:, 0]) - margin,
                  max(yhat[:, 0]) + margin,
                  min(yhat[:, 1]) - margin,
                  max(yhat[:, 1]) + margin])
        plt.pause(0.0001)

    @staticmethod
    def compute_visibility(observations, color):
        #observations = observations.astype(np.float)
        color_reshape = np.reshape(color, [1, 1, 3, 1, 1])

        observations = observations - color_reshape
        pixel_is_color = np.all(observations == 0, axis=2)

        values = np.sum(pixel_is_color, (2, 3))

        return values

    @staticmethod
    def getTransitionModel(b, m, dt):
        return np.array([[1,     0,     dt, 0 ],
                         [0,     1,     0,  dt],
                         [-dt/m, 0,     b,  0 ],
                         [0,     -dt/m, 0,  b ]])

    @staticmethod
    def getTransitionCovar(dyn_sig, useVar=True):
        fact = dyn_sig ** 2 if useVar else dyn_sig
        return fact * np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    @staticmethod
    def getObservationMatrix():
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])


def test():
    import matplotlib.pyplot as plt
    balls = Balls(
        n_balls=20,
        m=100,
        b=0.99,
        dyn_sig=0.002,
        w=64,
        h=64,
        data_size=5,
        cache=True,
        seed=0
    )
    balls.collect(150, 10)
 #   print('bla')
    #print(balls.y[0, :, :])
    #plt.figure()
    #plt.imshow(balls.o[0, 0, 1, :, :])
    #plt.show()
    #color = np.array([255, 0, 0])
    #obs = balls.o
    #colors = get_arrays(obs, color)
    #targets = balls.y

    #bla = p.plot_n_trajectories(2, targets[0, :, :], colors[0, :])
    #p.save_fig(fig=bla, path='BallsFig', name='traj')
    #plt.figure()
    balls.animate(seq_num=0, filename="/home/philipp/Code/test.avi")
    #print('bla')
    #for i in range(150):
    #    if balls.visible[i] == 0:
    #        plt.figure()
    #        plt.imshow()

def compute_visibility(observations, color):
    observations = observations.astype(np.float)
    color_reshape = np.reshape(color, [1, 1, 3, 1, 1])

    obs = observations - color_reshape
    pixel_is_color = np.all(obs == 0, axis=2)
    number_of_color_pixels = np.sum(pixel_is_color, (2, 3))

    partly_visible = np.any(pixel_is_color, axis=(2, 3))
    fully_visible = number_of_color_pixels == np.max(number_of_color_pixels)


    values = 2 * np.ones([observations.shape[0], observations.shape[1]])        # fully covered: 2
    values[partly_visible] = 1                                                  # partially visible: 1
    values[fully_visible] = 0                                                   # fully visible: 0

    return values

