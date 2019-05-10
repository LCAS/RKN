import tensorflow as tf
from network import HiddenLayersParamsKeys
from preprocessor.ImageNoisePreprocessor import ImageNoisePreprocessor
from preprocessor.ImageDropoutPreprocessor import ImageDropoutPreprocessor
from model import RKN
from model_local_linear.LLRKNConfig import LLRKNConfig
import numpy as np
""" See super class for documentation"""
class PendulumLLConfig(LLRKNConfig):

    NOISE_MODEL_WHITE = "white_noise"
    NOISE_MODEL_DROPOUT = "dropout"

    def __init__(self, img_size, output_mode, task_mode, init_mode, transition_noise_var,
                 use_likelihood, individual_sigma, noise_model, latent_observation_dim=2):
        assert output_mode == RKN.OUTPUT_MODE_POSITIONS or output_mode == RKN.OUTPUT_MODE_OBSERVATIONS, \
            "invalid output mode - needs to be either RKN.OUTPUT_MODE_POSITIONS or RKN.OUTPUT_MODE_OBSERVATIONS"
        assert task_mode == RKN.TASK_MODE_FILTER or RKN.TASK_MODE_PREDICT, \
            "invalid task mode - needs to be either RKN.TASK_MODE_FILTER or RKN.TASK_MODE_PREDICT"

      #  assert init_mode == RKNTransitionCell.INIT_MODE_LEARNED or init_mode == RKNTransitionCell.INIT_MODE_COPY_OBSERVATION, \
       #     "invalid init mode - need to bei either RKNTransitionCell.INIT_MODE_LEARNED or RKNTransitionCell.INIT_MODE_COPY_OBSERVATION"

        self._img_size = img_size
        self._task_mode = task_mode
        self._init_mode = init_mode
        self._use_likelihood = use_likelihood
        self._individual_sigma = individual_sigma

        self._latent_observation_dim = latent_observation_dim
        self._output_mode = output_mode

        if noise_model == PendulumLLConfig.NOISE_MODEL_WHITE:
            self.image_noise_preprocessor = ImageNoisePreprocessor(r=0.2) if self._task_mode == RKN.TASK_MODE_FILTER else None
        elif noise_model == PendulumLLConfig.NOISE_MODEL_DROPOUT:
            self.image_noise_preprocessor = ImageDropoutPreprocessor(2, 5, 5) if self._task_mode == RKN.TASK_MODE_FILTER else None
        else:
            raise AssertionError("Invalid Noise Model")

        #raise AssertionError("Should be learned")
        if transition_noise_var == "learn":
          init = np.log(1e-4)
          log_transition_noise_var_upper = tf.get_variable(name="log_sigma_trans_up", shape=[1,1],
                                                        initializer=tf.constant_initializer(init))
          log_transition_noise_var_lower = tf.get_variable(name="log_sigma_trans_lo", shape=[1,1],
                                                          initializer=tf.constant_initializer(init))
          self.transition_noise_var_upper = tf.exp(log_transition_noise_var_upper)
          self.transition_noise_var_lower = tf.exp(log_transition_noise_var_lower)
          self._transition_noise_var = tf.concat([tf.tile(self.transition_noise_var_upper, [1, self.latent_observation_dim]),
                            tf.tile(self.transition_noise_var_lower, [1, self.latent_observation_dim])], -1)
        else:
            cm = np.zeros([1, self.latent_state_dim])
            cm[self.latent_observation_dim:] = transition_noise_var
            self._transition_noise_var = tf.constant(cm, dtype=tf.float32)

    @property
    def input_dim(self):
        return self._img_size

    @property
    def output_dim(self):
        # 2 since we represent the state (angle) alpha as [sin(alpha), cos(alpha)]
        return 2 if self._output_mode == RKN.OUTPUT_MODE_POSITIONS else self._img_size

    @property
    def output_mode(self):
        return self._output_mode

    @property
    def latent_observation_dim(self):
        return self._latent_observation_dim

    @property
    def latent_state_dim(self):
        return 2 * self._latent_observation_dim

    @property
    def preprocessor(self):
        return self.image_noise_preprocessor

    '''Encoder'''

    @property
    def encoder_conv_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                HiddenLayersParamsKeys.PADDING: 'SAME',
                HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 3,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 12,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 2,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '1': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '1': 2,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '2': 2,
                HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '2': 2}

    @property
    def encoder_dense_dict(self):
        return  {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                 HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                 HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                 HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 50,
                 HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 25}

    '''TransitionModel'''
    @property
    def num_basis(self):
        return 16

    @property
    def transition_network_hidden_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 100}

    @property
    def transition_covariance(self):
        return self._transition_noise_var

    @property
    def init_mode(self):
        return self._init_mode

    @property
    def transition_matrix_mode(self):
        return RKN.INIT_TRANS_MATRIX_PURE_DIAG

    @property
    def normalize_latent(self):
        return True

    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 50,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '2': self.latent_state_dim}

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
    def use_likelihood(self):
        return self._use_likelihood

    @property
    def individual_sigma(self):
        return self._individual_sigma

    @property
    def variance_decoder_dict(self):
        return None # use linear mapping..

    '''Training'''
    @property
    def learning_rate(self):
        return 1e-3

    @property
    def reg_loss_factor(self):
        return 0.0

    @property
    def batch_size(self):
        return 50

    @property
    def bptt_length(self):
        if self._task_mode == RKN.TASK_MODE_FILTER:
            return 75
        elif self._task_mode == RKN.TASK_MODE_PREDICT:
            return 50
        else:
            raise AssertionError("Invalid task mode, has to be either RKN.TASK_MODE_FILTER or RKN.TASK_MODE_PREDICT")
