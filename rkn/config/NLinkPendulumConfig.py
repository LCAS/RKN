import tensorflow as tf
from network import HiddenLayersParamsKeys
from preprocessor.ImageNoisePreprocessor import ImageNoisePreprocessor
from model import RKNConfig, RKN
from transition_cell import TransitionCell
import numpy as np
""" See super class for documentation"""
class NLinkPendulumConfig(RKNConfig):

    VARIANCE_ACTIVATION_EXP = tf.exp
    VARIANCE_ACTIVATION_ELU = lambda x: tf.nn.elu(x) + 1

    def __init__(self,
                 name,
                 task_mode,
                 output_mode,
                 bw,
                 num_links,
                 latent_observation_dim,
                 model_type,
                 decoder_mode,
                 reg_loss_factor):

        super().__init__(name)


      #  assert init_mode == RKNTransitionCell.INIT_MODE_LEARNED or init_mode == RKNTransitionCell.INIT_MODE_COPY_OBSERVATION, \
       #     "invalid init mode - need to bei either RKNTransitionCell.INIT_MODE_LEARNED or RKNTransitionCell.INIT_MODE_COPY_OBSERVATION"

        self._img_size = [48, 48, 1]
        self._task_mode = task_mode
        self._output_mode = output_mode
        self._bw = bw
        self._num_links = num_links
        self._latent_observation_dim = latent_observation_dim
        self._model_type = model_type
        self._decoder_mode = decoder_mode
        self._reg_loss_factor = reg_loss_factor

    @property
    def input_dim(self):
        return self._img_size

    @property
    def output_dim(self):
        # 2 since we represent the state (angle) alpha as [sin(alpha), cos(alpha)]
        return (2 * self._num_links) if self._output_mode == RKN.OUTPUT_MODE_POSITIONS else self._img_size

    @property
    def output_mode(self):
        return self._output_mode

    @property
    def latent_observation_dim(self):
        return self._latent_observation_dim

    @property
    def latent_state_dim(self):
        return 2 * self.latent_observation_dim

    @property
    def preprocessor(self):
        return None

    @property
    def variance_activation_fn(self):
        return NLinkPendulumConfig.VARIANCE_ACTIVATION_ELU

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
        return 0

    @property
    def with_velocity_targets(self):
        return False

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
                HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 2,
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
    def single_encoder_variance(self):
        return False

    @property
    def max_encoder_variance(self):
        return -1

    @property
    def use_constant_observation_covariance(self):
        """Whether to learn a constant or input dependend variance"""
        return False

    @property
    def constant_observation_covariance_given(self):
        return False

    @property
    def constant_observation_covariance_init(self):
        return 0.1

    '''TransitionModel'''
    @property
    def bandwidth(self):
        return self._bw

    @property
    def num_basis(self):
      #  tf.logging.warn("CHECK NUMBER OF BASIS")
        return 15

    @property
    def transition_network_hidden_dict(self):
        """if self._hidden_transition_network:
            return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                    HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                    HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                    HiddenLayersParamsKeys.WIDTH_PREFIX + "1": 100}
        else:"""
        return None

    @property
    def init_mode(self):
        return RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE

    @property
    def transition_matrix_is_given(self):
        return False

    @property
    def transition_matrix_mode(self):
        """What kind of transition matrix to use and how to initialize it"""
        return RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH if self._model_type != "lstm" else RKN.TRANS_MATRIX_BAND

    @property
    def transition_matrix_init(self):
        return RKN.TRANS_INIT_FIX

    @property
    def transition_covariance_given(self):
        return False

    @property
    def transition_covar_hidden_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + "1": 10}

    @property
    def learn_correlated_transition_covar(self):
        return False

    @property
    def transition_covariance_init(self):
        return [1e-3, 1e-3, 0.0]

    @property
    def transition_covariance(self):
        return [0.1, 0.1, 0.0]

    @property
    def inidividual_transition_covar(self):
        return True

    @property
    def learn_state_dependent_transition_covar(self):
        return False

    @property
    def initial_state_covariance_given(self):
        return True

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

    @property
    def normalize_obs(self):
        """Whether to normalize latent observations, this should be true except when working with known
        models and/or true spaces"""
        return True #self._normalize_obs

    @property
    def normalize_prior(self):
        """Whether to normalize latent priors, this should be true except when working with known
        models and/or true spaces"""
        return False

    @property
    def normalize_posterior(self):
        """Whether to normalize the latent posterior, this should be true except when working with known
        models and/or true spaces"""
        return False

    @property
    def adapt_variance_to_normalization(self):
        return False

    @property
    def use_sigmoid_in_normalization(self):
        return False


    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 10}

    @property
    def decoder_dense_observation_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 144}

    @property
    def decoder_conv_dict(self):

        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                HiddenLayersParamsKeys.PADDING: 'SAME',
                HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 4,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 3,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 12,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 4}
    @property
    def decoder_initial_shape(self):
        return [3, 3, 16]
    @property
    def decoder_mode(self):
        return self._decoder_mode

    @property
    def decoder_matrix(self):
        return np.concatenate([np.eye(self.latent_observation_dim),
                               np.zeros([self.latent_observation_dim, self.latent_observation_dim])], -1)

    @property
    def fix_likelihood_decoder(self):
        return False #self._fix_likelihood_decoder or self.decoder_mode == RKN.DECODER_MODE_FIX

    @property
    def use_likelihood(self):
        return True #self._use_likelihood

    @property
    def individual_sigma(self):
        return self._output_mode == RKN.OUTPUT_MODE_POSITIONS

    @property
    def variance_decoder_dict(self):
        return self.decoder_dense_position_dict

    '''Training'''
    @property
    def learning_rate(self):
       # pc_rkn = lambda x: tf.train.piecewise_constant(x, boundaries=[800, 1600], values=[0.01, 0.005, 0.001])
       # pc_llrkn = lambda x: tf.train.piecewise_constant(x, boundaries=[400, 800, 1600], values=[0.05, 0.01, 0.005, 0.001])
        return 1e-3 #pc_llrkn if self._model_type == "llrkn" else (1e-3 if self._model_type == "lstm" else pc_rkn)

    @property
    def reg_loss_factor(self):
        return self._reg_loss_factor

    @property
    def batch_size(self):
        #if self._task_mode == RKN.TASK_MODE_FILTER:
        return 50

        #raise AssertionError("Currently only filtering") #Invalid task mode, has to be either RKN.TASK_MODE_FILTER or RKN.TASK_MODE_PREDICT")

    @property
    def bptt_length(self):
        #if self._task_mode == RKN.TASK_MODE_FILTER:
        return 75         #raise AssertionError("Currently only filtering") #Invalid task mode, has to be either RKN.TASK_MODE_FILTER or RKN.TASK_MODE_PREDICT")

    @property
    def scale_targets(self):
        return 255.0 if self._output_mode == RKN.OUTPUT_MODE_OBSERVATIONS else None