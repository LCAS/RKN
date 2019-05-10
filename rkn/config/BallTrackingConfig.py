import tensorflow as tf
from transition_cell.TransitionCell import TransitionCell
from model.RKNConfig import RKNConfig
from network.HiddenLayers import HiddenLayersParamsKeys
from transition_cell.RKNSimpleTransitionCell import RKNSimpleTransitionCell
from model import RKN
from data.LinearBalls import LinearBalls
import numpy as np

class BallTrackingConfig(RKNConfig):

    DYNAMICS_MODE_LINEAR = 'linear'
    DYNAMICS_MODE_DOUBLE_LINK = 'double_link'
    DYNAMICS_MODE_QUAD_LINK = 'quad_link'


    '''Main Model Parameters'''

    def __init__(self,
                 name,
                 dynamics_mode,
                 latent_obs_dim,
                 model_type,
                 transition_matrix,
                 give_linear_transition_matrix,
                 decoder_mode,
                 normalize_latent,
                 normalize_obs_only,
                 tm,
                 reg_loss_factor):

        super().__init__(name=name)

        assert dynamics_mode in [BallTrackingConfig.DYNAMICS_MODE_LINEAR,
                                 BallTrackingConfig.DYNAMICS_MODE_DOUBLE_LINK,
                                 BallTrackingConfig.DYNAMICS_MODE_QUAD_LINK], "Invalid Data mode"
        self._dynamics_mode = dynamics_mode
        self._model_type = model_type
        self._latent_obs_dim = latent_obs_dim
        self._transition_matrix = transition_matrix
        self._give_linear_transition_matrix = give_linear_transition_matrix
        self._decoder_mode = decoder_mode
        self._normalize_latent = normalize_latent
        self._normalize_obs_only = normalize_obs_only
        self._reg_loss_factor = reg_loss_factor

        self._high_dim_lin = False
        self._use_likelihood = False
        self._fix_likelihood_decoder = False #fix_likelihood_decoder
        self._individual_sigma = True #individual_sigma

        self._tm = tm


      #  assert dynamics_mode == BallTrackingHyperparameters.DYNAMICS_MODE_LINEAR, "currently only linear implemented"

    @property
    def input_dim(self):
        size = 64
        return [size, size, 3]

    @property
    def output_dim(self):
         return 2

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        if self._dynamics_mode == self.DYNAMICS_MODE_LINEAR:
            return 2
        else:
            return self._latent_obs_dim

    @property
    def latent_state_dim(self):
        return 2 * self.latent_observation_dim

    @property
    def variance_activation_fn(self):
        return lambda x: tf.nn.elu(x) + 1

    @property
    def never_invalid(self):
        return True

    @property
    def transition_cell_type(self):
        if self._model_type == 'rkns':
            return TransitionCell.TRANSITION_CELL_SIMPLE
        elif self._model_type == 'rknc':
            return TransitionCell.TRANSITION_CELL_CORRELATED
        elif self._model_type == 'llrkn':
            return TransitionCell.TRANSITION_CELL_CORRELATED
        else:
            return None

    '''Encoder'''
    @property
    def encoder_conv_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                HiddenLayersParamsKeys.PADDING: 'SAME',
                HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 4,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 1,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '1': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '1': 2,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 9,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 8,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '2': 2}

    @property
    def encoder_dense_dict(self):
        if self._dynamics_mode == self.DYNAMICS_MODE_LINEAR:
            return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                    HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                    HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 8,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 24}
        else:
            return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                    HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                    HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim}

    @property
    def constant_observation_covariance(self):
        return False

    @property
    def single_encoder_variance(self):
        return False

    @property
    def max_encoder_variance(self):
        return -1

    '''TransitionModel'''
    @property
    def bandwidth(self):
        if self._dynamics_mode == self.DYNAMICS_MODE_LINEAR:
            # Since latent observation dim = 2 this equals a full matrix
            return 2
        else:
            return 5
    """
    @property
    def num_basis(self):
        return 16

    @property
    def transition_network_hidden_dict(self):
        if self._hidden_transition_layer:
            return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                    HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                    HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                    HiddenLayersParamsKeys.WIDTH_PREFIX +"1": 100}
        else:
            return None 
    """
    @property
    def transition_matrix_init(self):
        return RKN.TRANS_INIT_RAND

    @property
    def init_mode(self):
        return RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE

    @property
    def transition_matrix_is_given(self):
        return self._give_linear_transition_matrix

    @property
    def transition_matrix(self):
        return self._tm

    @property
    def transition_matrix_mode(self):
        """What kind of transition matrix to use and how to initialize it"""
        return self._transition_matrix if self._model_type != "lstm" else RKN.TRANS_MATRIX_BAND

    @property
    def transition_covariance_given(self):
        return self._give_linear_transition_matrix

    @property
    def inidividual_transition_covar(self):
        return True

    @property
    def transition_covariance(self):
        return [1e-10, (2 * 1e-3) ** 2, 0.0]

    @property
    def transition_covariance_init(self):
        return [1e-3, 1e-3, 0.0]


    @property
    def initial_state_covariance_given(self):
        return False

    @property
    def initial_state_covariance_init(self):
        """Value to initialize the initial state covariance in case its learned"""
        return 1.0

    @property
    def normalize_obs(self):
        if self._dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR:
            return False
        else:
            return self._normalize_obs_only

    @property
    def normalize_posterior(self):
        return not self._normalize_obs_only

    @property
    def normalize_prior(self):
        return not self._normalize_obs_only

    @property
    def normalize_latent(self):
        if self.transition_matrix_is_given:
            return False
        else:
            return self._normalize_latent

    @property
    def adapt_variance_to_normalization(self):
        return False

    '''Decoder'''

    @property
    def decoder_dense_position_dict(self):
        if self._dynamics_mode == self.DYNAMICS_MODE_LINEAR:
            if self._high_dim_lin:
                return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                        HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                        HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                        HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim}
            else:
                raise AssertionError("It appears decoder should be trained despite using linear dynamics")
        else:
            return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                    HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                    HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 10}



    @property
    def decoder_mode(self):
        if self._dynamics_mode == self.DYNAMICS_MODE_LINEAR and not self._high_dim_lin:
            return RKN.DECODER_MODE_FIX
        else:
            return self._decoder_mode

    @property
    def decoder_matrix(self):
        # Need to give the transpose here since we work with batches of row vectors
        #(We compute s^T=z^T H^T instead of s = Hz)
        return np.eye(2) #LinearBalls.observationMatrix()

    """Likelihood training"""
    @property
    def use_likelihood(self):
        """If this is true the gaussian likelihood is maximized instead of using RMSE/BCE.
         (Obviously the likelihood is maximized by minimizing the negative log likelihood)"""
        return self._use_likelihood

    @property
    def fix_likelihood_decoder(self):
        return self._fix_likelihood_decoder

    @property
    def individual_sigma(self):
        return self._individual_sigma

    @property
    def variance_decoder_dict(self):
        return self.decoder_dense_position_dict

    '''Training'''
    @property
    def learning_rate(self):
        return 1e-3

    @property
    def reg_loss_factor(self):
        return self._reg_loss_factor


    @property
    def batch_size(self):
        return 50 # if self._dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else 50

    @property
    def bptt_length(self):
        return 50 # if self._dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else 50
