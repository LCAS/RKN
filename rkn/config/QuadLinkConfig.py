import tensorflow as tf
from network.HiddenLayers import HiddenLayersParamsKeys
from model.RKN import RKN
from model.RKNConfig import RKNConfig
from preprocessor.GaussianNoisePreprocessor import GaussianNoisePreprocessor
from transition_cell.RKNSimpleTransitionCell import RKNSimpleTransitionCell

class QuadLinkConfig(RKNConfig):
    """Abstract class containing config for the RKN model - implement this for your own model"""

    '''Main Model Parameters'''
    def __init__(self, latent_obs_dim, obs_noise_std, use_log_likelihood, individual_sigma, batch_size,
                 bptt_length):
        self._latent_obs_dim = latent_obs_dim
        self._obs_preprocessor = GaussianNoisePreprocessor(obs_noise_std)
        self._use_log_likelihood = use_log_likelihood
        self._individual_sigma = individual_sigma
        self._batch_size = batch_size
        self._bptt_length = bptt_length


    @property
    def input_dim(self):
        return 2

    @property
    def output_dim(self):
        return 2

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        """Dimensionality of the latent observations (In Paper: m)"""
        return self._latent_obs_dim

    @property
    def latent_state_dim(self):
        """Dimensionality of the latent state (In Paper: n)"""
        return 2 * self.latent_observation_dim

    '''Preprocessor'''

    @property
    def preprocessor(self):
        return self._obs_preprocessor

    '''Encoder'''

    @property
    def encoder_conv_dict(self):
        raise NotImplementedError("Encoder conv dict not given")

    @property
    def encoder_dense_dict(self):
        #Todo tune this!
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 20,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 40}

    @property
    def only_dense_encoder(self):
        return True

    '''TransitionModel'''

    @property
    def bandwidth(self):
        #Todo: Tune
        return 3

    @property
    def transition_covariance(self):
        return 0.002 # it helps to add a bit of artificial noise

    @property
    def init_mode(self):
        return RKNSimpleTransitionCell.INIT_MODE_COPY_OBSERVATION

    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        #Todo Tune:
        return {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 20,
                HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 40}


    """Likelihood training"""
    @property
    def use_likelihood(self):
        return self._use_log_likelihood

    @property
    def individual_sigma(self):
        return self._individual_sigma

    @property
    def variance_decoder_dict(self):
        # Not Sure if this is clever yet...
        return self.decoder_dense_position_dict

    '''Training'''

    @property
    def learning_rate(self):
        return 1e-3
    @property
    def reg_loss_factor(self):
        return 0.0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def bptt_length(self):
        return self._bptt_length
