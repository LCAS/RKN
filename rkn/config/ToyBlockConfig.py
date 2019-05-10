from model.RKN import RKN
from transition_cell.TransitionCell import TransitionCell
from network.HiddenLayers import HiddenLayersParamsKeys
import tensorflow as tf
from model.RKNConfig import RKNConfig

class ToyBlockTowerConfig(RKNConfig):

    def __init__(self, name, output_mode, decoder_mode, latent_observation_dim=150):
        assert output_mode == RKN.OUTPUT_MODE_POSITIONS or output_mode == RKN.OUTPUT_MODE_OBSERVATIONS,\
        "Invalid output mode"

        super().__init__(name)

        self._output_mode = output_mode
        self._decoder_mode = decoder_mode
        print("LATENT_OBS_DIM", latent_observation_dim)
        self._latent_observation_dim = latent_observation_dim


    '''Main Model Parameters'''
    @property
    def input_dim(self):
        # If you change this the decoder (and probably the encoder too) need to be adapted
        return [60, 80, 3]

    @property
    def output_dim(self):
        if self._output_mode == RKN.OUTPUT_MODE_POSITIONS:
            return 9 # 3 * 3 position values (since we have 3 boxes)
        else: #self._output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
            return self.input_dim

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
    def variance_activation_fn(self):
        return lambda x: tf.nn.elu(x) + 1

    @property
    def transition_cell_type(self):
        return TransitionCell.TRANSITION_CELL_CORRELATED

    @property
    def n_step_pred(self):
        return -1

    @property
    def never_invalid(self):
        return True


    '''Encoder'''

    @property
    def encoder_conv_dict(self):
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 3,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
            HiddenLayersParamsKeys.PADDING: 'same',
            HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 3,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 1,
            HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '1': 2,
            HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '1': 2,

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 5,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 3,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 1,
            HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '2': 2,
            HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '2': 2,

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '3': 3,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '3': 3,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '3': 1,
            HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '3': 2,
            HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '3': 2}

    @property
    def encoder_dense_dict(self):
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 1,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
            HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim}

    @property
    def max_encoder_variance(self):
        return -1

    @property
    def single_encoder_variance(self):
        return False

    @property
    def use_constant_observation_covariance(self):
        return False

    '''TransitionModel'''

    @property
    def bandwidth(self):
        return 3

    @property
    def init_mode(self):
        return RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE

    @property
    def transition_matrix_mode(self):
        #tf.logging.warn("Check Transition matrix type")
        return RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH

    @property
    def transition_matrix_init(self):
        #tf.logging.warn("Check Transition Matrix init")
        return RKN.TRANS_INIT_RAND

    @property
    def transition_covariance(self):
        return 0.005

    @property
    def transition_covariance_given(self):
        return False

    @property
    def learn_correlated_transition_covar(self):
        return False

    @property
    def learn_state_dependent_transition_covar(self):
        return False

    @property
    def inidividual_transition_covar(self):
        return False

    @property
    def transition_covariance_init(self):
        #tf.logging.warn("Check Transition Covariance Init")
        return [0.1, 0.1, 0.0]

    @property
    def initial_state_covariance_init(self):
        return 1.0

    @property
    def normalize_latent(self):
        #tf.logging.warn("Check normalize latent")
        return False

    @property
    def adapt_variance_to_normalization(self):
        return False

    '''Decoder'''

    @property
    def decoder_dense_position_dict(self):
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 2,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
            HiddenLayersParamsKeys.WIDTH_PREFIX + '1': self.latent_observation_dim,
            HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 20}

    @property
    def decoder_dense_observation_dict(self):
        chan1 = self.decoder_conv_dict[HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1']
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 1,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
            HiddenLayersParamsKeys.WIDTH_PREFIX + '1': (5 ** 2) * chan1}

    @property
    def decoder_conv_dict(self):
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 4,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
            HiddenLayersParamsKeys.PADDING: 'same',

            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 15,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '1': [1, 2], #[2, 6],

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 5,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 15,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '2': [2, 2], #4, # [4, 3],

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '3': 5,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '3': 15,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '3': [3, 2], #4, # [3, 3],

            HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '4': 3,
            HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '4': 30,
            HiddenLayersParamsKeys.STRIDES_PREFIX + '4': [2, 2]}

    @property
    def decoder_initial_shape(self):
        chan1 = self.decoder_conv_dict[HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1']
        return [5, 5, chan1]

    @property
    def decoder_mode(self):
        return self._decoder_mode if self.output_mode == RKN.OUTPUT_MODE_POSITIONS else RKN.DECODER_MODE_NONLINEAR
    '''Training'''

    @property
    def use_likelihood(self):
        return True

    @property
    def fix_likelihood_decoder(self):
        return False

    @property
    def individual_sigma(self):
        return self.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS

    @property
    def variance_decoder_dict(self):
        return self.decoder_dense_position_dict

    @property
    def learning_rate(self):
        return 1e-3

    @property
    def reg_loss_factor(self):
        return 0

    @property
    def batch_size(self):
        return 50

    @property
    def bptt_length(self):
        return 15

    @property
    def scale_targets(self):
        return 255.0