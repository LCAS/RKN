import tensorflow as tf
from model.RKN import RKN
from baselines.GRUTransitionCell import GRUTransitionCell
class GRUBaseline(RKN):

    def _build_transition_cell(self):
        print("using GRU Transition Cell")
        self.transition_cell = GRUTransitionCell(2 * self.latent_state_dim)
        self.transition_cell.set_state_dim(self.latent_state_dim)

    def _build_initial_state(self):
        self.initial_latent_state = tf.placeholder(shape=[None,  2 *self.latent_state_dim], dtype=tf.float32)