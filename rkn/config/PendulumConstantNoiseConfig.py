import numpy as np
from model.RKN import RKN
from transition_cell.TransitionCell import TransitionCell
from network import HiddenLayersParamsKeys
import tensorflow as tf
from model_local_linear.LLRKNConfig import LLRKNConfig
class PendulumConstantNoiseConfig(LLRKNConfig):
    """Abstract class containing config for the RKN model - implement this for your own model"""

    def __init__(self,
                 name,
                 task_mode,
                 latent_observation_dim,
                 model_type,
                 transition_matrix,
                 use_likelihood,
                 fix_likelihood_decoder,
                 use_constant_observation_covariance,
                 state_dependent_transition_covar,
                 individual_transition_covar,
                 decoder_mode,
                 n_step_prediction,
                 adapt_var_to_norm,
                 use_sigmoid_in_normalization):
        super().__init__(name)

        self._task_mode = task_mode
        self._latent_observation_dim = latent_observation_dim
        self._model_type = model_type
        self._use_constant_observation_covariance = use_constant_observation_covariance
        self._use_likelihood = use_likelihood
        self._fix_liklehood_decoder = fix_likelihood_decoder
        self._transition_matrix = transition_matrix
        self._decoder_mode = decoder_mode
        self._state_dependent_transition_covar = state_dependent_transition_covar
        self._individual_transition_covar = individual_transition_covar
        self._adapt_var_to_norm = adapt_var_to_norm
        self._use_sigmoid_in_normalization = use_sigmoid_in_normalization
        self._n_step_pred = n_step_prediction

    '''Main Model Parameters'''

    @property
    def input_dim(self):
        return [24, 24, 1]

    @property
    def output_dim(self):
        return 2

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        return 2 if self._model_type == "llrkn" else self._latent_observation_dim

    @property
    def latent_state_dim(self):
        return 2 * self.latent_observation_dim

    @property
    def variance_activation_fn(self):
        return lambda x: tf.nn.elu(x) + 1

    @property
    def never_invalid(self):
        return self._task_mode == RKN.TASK_MODE_FILTER

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

    @property
    def n_step_pred(self):
        return self._n_step_pred

    '''Encoder'''
    @property
    def encoder_conv_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                HiddenLayersParamsKeys.PADDING: 'SAME',
                HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 12,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 1,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '1': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '1': 2,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 3,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 12,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '2': 2}


    @property
    def encoder_dense_dict(self):
        return  {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                 HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                 HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                 HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_state_dim}

    @property
    def max_encoder_variance(self):
        return -1

    @property
    def single_encoder_variance(self):
        return False

    @property
    def use_constant_observation_covariance(self):
        """Whether to learn a constant or input dependend variance"""
        return self._use_constant_observation_covariance

    @property
    def constant_observation_covariance(self):
        return [0.1]

    @property
    def constant_observation_covariance_given(self):
        return False

    @property
    def constant_observation_covariance_init(self):
        return 0.1

    '''TransitionModel'''

    @property
    def bandwidth(self):
        return 3

    @property
    def num_basis(self):
        return 16

    @property
    def transition_network_hidden_dict(self):
        return None
        #if self._hidden_transition_layer:
        #    return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
        #            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
        #            HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
        #            HiddenLayersParamsKeys.WIDTH_PREFIX +"1": 100}
        #else:
        #    return None

    @property
    def transition_matrix_init(self):
        return RKN.TRANS_INIT_FIX if self._model_type == "llrkn" else RKN.TRANS_INIT_RAND

    @property
    def init_mode(self):
        return RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE

    @property
    def transition_matrix_is_given(self):
        return False

    @property
    def transition_matrix_mode(self):
        """What kind of transition matrix to use and how to initialize it"""
        return self._transition_matrix if self._model_type != "lstm" else RKN.TRANS_MATRIX_BAND

    @property
    def transition_covariance_given(self):
        return False

    @property
    def learn_correlated_transition_covar(self):
        return False

    @property
    def transition_covariance_init(self):
        return [0.1, 0.1, 0.0]

    @property
    def transition_covariance(self):
        return [0.1, 0.1, 0.0]

    @property
    def inidividual_transition_covar(self):
        return self._individual_transition_covar

    @property
    def learn_state_dependent_transition_covar(self):
        return self._state_dependent_transition_covar

    @property
    def transition_covar_hidden_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + "1": 10}

    @property
    def initial_state_covariance_given(self):
        return False

    @property
    def initial_state_covariance(self):
        return 1.0

    @property
    def initial_state_covariance_init(self):
        """Value to initialize the initial state covariance in case its learned"""
        return 1.0

    @property
    def normalize_latent(self):
        return False
        #return self.transition_matrix_mode == RKN.TRANS_MATRIX_BAND

    @property
    def adapt_variance_to_normalization(self):
        return self._adapt_var_to_norm

    @property
    def use_sigmoid_in_normalization(self):
        return self._use_sigmoid_in_normalization
    '''Decoder'''

    @property
    def decoder_dense_position_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 10}

    @property
    def decoder_mode(self):
        return RKN.DECODER_MODE_FIX if self._model_type == "llrkn" else self._decoder_mode

    @property
    def decoder_matrix(self):
        return np.concatenate([np.eye(self.latent_observation_dim),
                               np.zeros([self.latent_observation_dim, self.latent_observation_dim])], -1)

    """Likelihood training"""
    @property
    def use_likelihood(self):
        """If this is true the gaussian likelihood is maximized instead of using RMSE/BCE.
         (Obviously the likelihood is maximized by minimizing the negative log likelihood)"""
        return self._use_likelihood

    @property
    def fix_likelihood_decoder(self):
        return self._fix_liklehood_decoder or self.decoder_mode == RKN.DECODER_MODE_FIX

    @property
    def individual_sigma(self):
        return True

    @property
    def variance_decoder_dict(self):
        return self.decoder_dense_position_dict
    '''Training'''

    @property
    def learning_rate(self):
        #pc_rkn = lambda x: tf.train.piecewise_constant(x, boundaries=[800, 1600], values=[0.01, 0.005, 0.001])
        #pc_llrkn = lambda x: tf.train.piecewise_constant(x, boundaries=[400, 800, 1600], values=[0.05, 0.01, 0.005, 0.001])
        #return pc_llrkn if self._model_type == "llrkn" else (1e-3 if self._model_type == "lstm" else pc_rkn)
        return 1e-3

    @property
    def reg_loss_factor(self):
        return 1.0

    @property
    def batch_size(self):
        return 50 if self._task_mode == RKN.TASK_MODE_FILTER else 15

    @property
    def bptt_length(self):
        return 75 if self._task_mode == RKN.TASK_MODE_FILTER else 50
