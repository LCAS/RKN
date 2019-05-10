from model.RKN import RKN
import tensorflow as tf
from baselines.LSTMTransitionCell import LSTMTransitionCell

class LSTMBaseline(RKN):

    def _build_transition_cell(self):
        print("using LSTM Transition Cell")
        self.transition_cell = LSTMTransitionCell(2 * self.latent_state_dim)
        self.transition_cell.set_state_dim(self.latent_state_dim)

    def _build_initial_state(self):
        self.c_init = tf.placeholder(shape=[None, 2 * self.latent_state_dim], dtype=tf.float32)
        self.m_init = tf.placeholder(shape=[None, 2 * self.latent_state_dim], dtype=tf.float32)

        self.initial_latent_state = tf.nn.rnn_cell.LSTMStateTuple(self.c_init, self.m_init)

