from model import RKNConfig
import numpy as np
from model import RKN
import tensorflow as tf
from network.HiddenLayers import HiddenLayersParamsKeys


class KittiConfig(RKNConfig):
    """Abstract class containing config for the RKN model - implement this for your own model"""

    def __init__(self, latent_observation_dim, bandwidth, init_mode, batch_size, seq_length, transition_noise_covar):
        self._latent_observation_dim = latent_observation_dim
        self._bandwidth = bandwidth
        self._init_mode = init_mode
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._transition_noise_covar = transition_noise_covar

    '''Main Model Parameters'''

    @property
    def input_dim(self):
        return [50, 150, 6]

    @property
    def output_dim(self):
        return 3

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        return self._latent_observation_dim

    @property
    def latent_state_dim(self):
        return self._latent_observation_dim * 2

    '''Encoder'''
    @property
    def encoder_conv_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 4,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                HiddenLayersParamsKeys.PADDING: 'SAME',
                HiddenLayersParamsKeys.POOL: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX +  '1': 7,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX +      '1': 1,

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX +  '2': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX +      '2': [1, 2],

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX +  '3': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '3': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX +      '3': [1, 2],

                HiddenLayersParamsKeys.FILTER_SIZE_PREFIX +  '4': 5,
                HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '4': 16,
                HiddenLayersParamsKeys.STRIDES_PREFIX +      '4': [2, 2]}

    @property
    def encoder_dense_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 128,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 128}

    '''TransitionModel'''
    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def transition_covariance(self):
        cm = np.zeros([1, self.latent_state_dim])
        cm[self.latent_observation_dim:] = self._transition_noise_covar
        return tf.constant(cm, tf.float32)


    @property
    def init_mode(self):
       return self._init_mode

    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '2': int(self.latent_observation_dim / 2)}

    '''Training'''
    @property
    def learning_rate(self):
        return 1e-3

    @property
    def reg_loss_factor(self):
        return 0.0

    @property
    def batch_size(self):
        """Batch size during training"""
        return self._batch_size

    @property
    def bptt_length(self):
        """Steps before gradients are truncated for truncated Backprop Trough Time"""
        return self._seq_length