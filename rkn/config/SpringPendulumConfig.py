import tensorflow as tf
from model.RKN import RKN
from model.RKNConfig import RKNConfig
import numpy as np

class SpringPendulumConfig(RKNConfig):
    """Abstract class containing config for the RKN model - implement this for your own model"""

    def __init__(self,
                 name,
                 dim,
                 transition_cell_type,
                 use_likelihood,
                 transition_matrix=None,
                 transition_covariance=None,
                 transition_covariance_init=None,
                 initial_state_covariance=None,
                 initial_state_covariance_init=None,
                 ):
        super().__init__(name)

        assert transition_covariance is None or transition_covariance_init is None, \
            "Both transition covariance and initial value for learning given"
        assert (not transition_covariance is None) or (not transition_covariance_init is None), \
            "Neither transition covariance nor initial value for learning given"

        assert initial_state_covariance is None or initial_state_covariance_init is None, \
            "Both initial state covariance and initial value for learning given"
        assert (not initial_state_covariance is None) or (not initial_state_covariance_init is None), \
            "Neither initial state covariance nor initial value for learning given"

        self._transition_matrix = transition_matrix
        self._transition_covariance = transition_covariance
        self._transition_covariance_init = transition_covariance_init
        self._initial_state_covariance = initial_state_covariance
        self._initial_state_covariance_init = initial_state_covariance_init

        self._transition_cell_type = transition_cell_type
        self._dim = dim
        self._use_likelihood = use_likelihood

    '''Main Model Parameters'''

    @property
    def input_dim(self):
        return self._dim

    @property
    def output_dim(self):
        return self._dim

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        return self._dim

    @property
    def latent_state_dim(self):
        return self._dim * 2

    @property
    def variance_activation_fn(self):
        return tf.exp

    @property
    def never_invalid(self):
        return True

    '''TransitionModel'''

    @property
    def transition_cell_type(self):
        return self._transition_cell_type

    @property
    def transition_matrix_is_given(self):
        return self._transition_matrix is not None

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def transition_covariance_given(self):
        return self._transition_covariance is not None

    @property
    def transition_covariance(self):
        return self._transition_covariance

    @property
    def transition_covariance_init(self):
        return self._transition_covariance_init

    @property
    def initial_state_covariance_given(self):
        return self._initial_state_covariance is not None

    @property
    def initial_state_covariance(self):
        return self._initial_state_covariance

    @property
    def initial_state_covariance_init(self):
        return self._initial_state_covariance_init

    @property
    def normalize_latent(self):
        return False

    @property
    def use_likelihood(self):
        return self._use_likelihood

    '''Training'''

    @property
    def learning_rate(self):
        return 1e-2

    @property
    def batch_size(self):
        return 10

    @property
    def bptt_length(self):
       return 100
