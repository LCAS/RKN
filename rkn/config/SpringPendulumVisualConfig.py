from model import RKNConfig, RKN
from transition_cell import TransitionCell
import tensorflow as tf
import numpy as np
from network.HiddenLayers import HiddenLayersParamsKeys

class SpringPendulumVisualConfig(RKNConfig):

    def __init__(self,
                 name,
                 observation_dim,
                 transition_cell_type,
                 transition_matrix=None,
                 transition_covar=None,
                 transition_covar_init=None,
                 observation_covar=None,
                 observation_covar_init=None,
                 initial_covar=None,
                 initial_covar_init=None,
                 model_type=None):
        super().__init__(name)
        self._observation_dim = observation_dim
        self._transition_cell_type = transition_cell_type

        self._transition_matrix = transition_matrix

        self._check_input(transition_covar, transition_covar_init, "transition")
        self._transition_covar = transition_covar
        self._transition_covar_init = transition_covar_init

        self._check_input(observation_covar, observation_covar_init, "observation")
        self._observation_covar = observation_covar
        self._observation_covar_init = observation_covar_init

        self._check_input(initial_covar, initial_covar_init, "initial")
        self._initial_covar =initial_covar
        self._initial_covar_init = initial_covar_init

        self._model_type = model_type

    def _check_input(self,a, b, msg):
        assert (a is None and not b is None) or (b is None and not a is None), "Val/Init invalid for, " + msg

    @property
    def input_dim(self):
        return self._observation_dim

    @property
    def output_dim(self):
        return 1

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        return 5

    @property
    def latent_state_dim(self):
        return 10

    @property
    def variance_activation_fn(self):
        return tf.exp

    @property
    def never_invalid(self):
        return True

    @property
    def transition_cell_type(self):
        return self._transition_cell_type

    '''Preprocessor'''

    @property
    def preprocessor(self):
        """Preprocessor to apply on the input data before it is given to the model.
        Needs to be a callable object taking two parameters - a batch of sequences of images (i.e.
        a 5 D tensor) and a flag indicating whether the code is run on gpu or not
        (and hence if the image format is CWH or WHC)
        """
        return None

    '''Encoder'''

    @property
    def encoder_dense_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
               # HiddenLayersParamsKeys.KEEP_PROB_PREFIX + "1": 0.8,
                HiddenLayersParamsKeys.WIDTH_PREFIX + "1": 10}

    @property
    def only_dense_encoder(self):
        return True

    @property
    def max_encoder_variance(self):
        return -1

    @property
    def single_encoder_variance(self):
        return True

    @property
    def use_constant_observation_covariance(self):
        """Whether to learn a constant or input dependend variance"""
        return True

    @property
    def constant_observation_covariance_given(self):
        """If constant covariance is used, whether it is given or learned"""
        return self._observation_covar is not None

    @property
    def constant_observation_covariance(self):
        """The constant observation covar, if its given"""
        return self._observation_covar

    @property
    def constant_observation_covariance_init(self):
        """Value to initialize the constant covariance with if its given"""
        return self._observation_covar_init

    '''TransitionModel'''

    @property
    def init_mode(self):
        return RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE

    @property
    def transition_matrix_is_given(self):
        """Whether the transition matrix is given - if true transition_matrix needs to return the (true) transition
          matrix"""
        return self._transition_matrix is not None

    @property
    def transition_matrix(self):
        """The (true) transition matrix - only relevant if transition_matrix_given"""
        return self._transition_matrix

    @property
    def transition_matrix_mode(self):
        """What kind of transition matrix to use and how to initialize it"""
        return RKN.TRANS_MATRIX_STABLE_SPRING_DAMPER

    @property
    def transition_covariance_given(self):
        """Whether the transition noise is given - if true transition_noise_covar needs to return the (true) transition
        noise covariance"""
        return self._transition_covar is not None

    @property
    def transition_covariance(self):
        """True transition noise covariance in case its given"""
        return self._transition_covar

    @property
    def transition_covariance_init(self):
        """Value to initialize the transition covariance in case its learned"""
        return self._transition_covar_init

    @property
    def initial_state_covariance_given(self):
        return self._initial_covar is not None

    @property
    def initial_state_covariance(self):
        """Initial state covariance to use if given"""
        return self._initial_covar

    @property
    def initial_state_covariance_init(self):
        """Value to initialize the initial state covariance in case its learned"""
        return self._initial_covar_init

    @property
    def normalize_latent(self):
        """Whether to normalize latent_states and observations, this should be true except when working with known
        models and/or true spaces"""
        return False

    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 5}

    @property
    def decoder_mode(self):
        return RKN.DECODER_MODE_NONLINEAR if self._model_type == "lstm" else RKN.DECODER_MODE_FIX

    @property
    def decoder_matrix(self):
        """Matrix to use as decoder - only relevant if decoder_is_fixed"""
        return np.array([[1, 0]])

    """Likelihood training"""

    @property
    def use_likelihood(self):
        """If this is true the gaussian likelihood is maximized instead of using RMSE/BCE.
         (Obviously the likelihood is maximized by minimizing the negative log likelihood)"""
        return True

    @property
    def fix_likelihood_decoder(self):
        """If true the variance of the output is computed as the mean of the latent state variance, only works if
        individual_sigma is false"""
        return True

    @property
    def individual_sigma(self):
        """If true learn one variance value for each entry of the output, if false learns a mutual value for all entries,
        should be true if outputs are images"""
        return False

    @property
    def variance_decoder_dict(self):
        return self.decoder_dense_position_dict
    '''Training'''

    @property
    def learning_rate(self):
        """Learning rate of the Adam Optimizer (\alpha)"""
        return 1e-3
    @property
    def reg_loss_factor(self):
        """Scaling factor of the L2 regularization loss"""
        return 0.0

    @property
    def batch_size(self):
        return 50

    @property
    def bptt_length(self):
        return 100