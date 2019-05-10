import numpy as np
import _QuadLinkForwardModel as QuadLink

class QuadLinkData:

    def __init__(self, episode_length, train_episodes, test_episodes, generate_actions=False, **kwargs):

        # simulation parameters
        self.maxVelo = kwargs['maxVelo'] if 'maxVelo' in kwargs else 8
        self.maxTorque = kwargs['maxTorque'] if 'maxTorque' in kwargs else 10
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1e-2
        self.masses = kwargs['mass'] if 'mass' in kwargs else np.ones(4)
        self.lengths = kwargs['length'] if 'length' in kwargs else np.ones(4)
        self.inertias = self.masses * self.lengths**2 / 3
        self.g = kwargs['g'] if 'g' in kwargs else 9.81
        self.friction = kwargs['friction'] if 'friction' in kwargs else np.zeros(4)
        self.sim_dt = kwargs['sim_dt'] if 'sim_dt' in kwargs else 1e-4

        # generation parameters
        self.generate_actions = generate_actions
        self.simulation_steps = episode_length
        self.nr_epochs_train = train_episodes
        self.nr_epochs_test = test_episodes

        print('generating sets: epoch_length:', episode_length,
              'training epochs:', train_episodes,
              'testing epochs:', test_episodes)

        self.x_train, self.u_train, self.s_train =\
            self._create_data_set(self.nr_epochs_train, self.simulation_steps + 1)
        self.x_test, self.u_test, self.s_test =\
            self._create_data_set(self.nr_epochs_test, self.simulation_steps + 1)

        self.current_index = 0

    def get_batch(self, idx):
        return self.x_train[:, idx, :], self.u_train[:, idx, :], self.x_train[:, idx + 1, :]

    def get_init_train(self):
        return self.x_train[:, 0, :]

    def get_train_data(self, ts=False):
        if ts:
            return self.x_train[:, :self.simulation_steps, :], \
                   self.u_train,\
                   self.x_train[:, 1:, :],\
                   self.s_train[:, 1:, :]
        else:
            return self.x_train[:, :self.simulation_steps, :], \
                   self.u_train,\
                   self.x_train[:, 1:, :],\

    def get_test_data(self, ts=False):
        if ts:
            return self.x_test[:, :self.simulation_steps, :], \
                   self.u_test, \
                   self.x_test[:, 1:, :],\
                   self.s_test[:, 1:, :]
        else:
            return self.x_test[:, :self.simulation_steps, :], \
                   self.u_test, \
                   self.x_test[:, 1:, :],\


    def _transitionFunction(self, states, actions):
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


    def _sampleAction(self, shape):
        if(self.generate_actions):
            return np.random.uniform(-self.maxTorque, self.maxTorque, shape)
        else:
            return np.zeros(shape=shape)

    # states in form [p0, v0, p1, v1, p2,v2, p3, v3]
    def _generateEndefectorPositions(self, states):
        positions = np.zeros((states.shape[0], 2))
        for i in range(4):
            positions[:, 0] += np.cos(states[:, 2 * i]) * self.lengths[i]
            positions[:, 1] += np.sin(states[:, 2 * i]) * self.lengths[i]
        return positions

    def _sample_init_states(self, nr_epochs):
        init_state = np.zeros((nr_epochs, 8))
        init_pos = np.random.uniform(-np.pi, np.pi, (nr_epochs, 4))
        for i in range(4):
            init_state[:, 2 * i] = init_pos[:, i]
        return init_state

    def _create_data_set(self, nr_epochs, epoch_length):

        states = np.zeros((nr_epochs, epoch_length, 8))
        actions = self._sampleAction((nr_epochs, epoch_length, 4))
        endefector_pos = np.zeros((nr_epochs, epoch_length, 2))

        states[:, 0, :] = self._sample_init_states(nr_epochs)
        endefector_pos[:, 0, :] = self._generateEndefectorPositions(states[:, 0, :])

        for i in range(1, epoch_length):
            states[:, i, :] = self._transitionFunction(states[:, i - 1, :], actions[:, i - 1, :])
            for j in range(0, nr_epochs):
                for k in range(0, 4):
                    if states[j, i, 2 * k] > np.pi:
                        states[j, i, 2 * k] -= 2 * np.pi
                    if states[j, i, 2 * k] < -np.pi:
                        states[j, i, 2 * k] += 2 * np.pi
            endefector_pos[:, i, :] = self._generateEndefectorPositions(states[:, i, :])

        return endefector_pos, actions, states


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ql_data = QuadLinkData(150, 10, 10, dt=0.05)

    pos, _, _ = ql_data.get_train_data()

    print("dummy")

    for i in range(10):
        plt.figure()
        plt.plot(pos[i, :, 0], pos[i, :, 1])
    plt.show()


