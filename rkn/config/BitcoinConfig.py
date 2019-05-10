from model.RKNConfig import RKNConfig
from model.RKN import RKN
from transition_cell.RKNSimpleTransitionCell import RKNSimpleTransitionCell
from network.HiddenLayers import HiddenLayersParamsKeys

import tensorflow as tf

class BitcoinConfig(RKNConfig):


    def __init__(self, num_features, latent_observation_dim, batch_size, bptt_length, loss_fn):

        self._loss_fn = loss_fn
        self._input_dim = num_features
        self._latent_observation_dim = latent_observation_dim
        self._batch_size = batch_size
        self._bptt_length = bptt_length

    '''Main Model Parameters'''

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return 1

    @property
    def output_mode(self):
        return RKN.OUTPUT_MODE_OBSERVATIONS if self._loss_fn == "bce" else RKN.OUTPUT_MODE_POSITIONS

    @property
    def latent_observation_dim(self):
        return self._latent_observation_dim

    @property
    def latent_state_dim(self):
        return 2 * self._latent_observation_dim

    '''Encoder'''

    @property
    def encoder_dense_dict(self):
        return {
            HiddenLayersParamsKeys.NUM_LAYERS: 1,
            HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
            HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
            HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 20,
            HiddenLayersParamsKeys.KEEP_PROB_PREFIX + '1': 0.5}

    @property
    def only_dense_encoder(self):
        return True

    '''TransitionModel'''

    @property
    def bandwidth(self):
        return 3

    @property
    def transition_covariance(self):
        return 0.01

    @property
    def init_mode(self):
        return RKNSimpleTransitionCell.INIT_MODE_LEARNED

    @property
    def normalize_latent(self):
        return True

    '''Decoder'''

    @property
    def decoder_dense_position_dict(self):
        return self.encoder_dense_dict

    @property
    def decoder_dense_observation_dict(self):
        return self.encoder_dense_dict

    @property
    def decoder_conv_dict(self):
        return None

    @property
    def decoder_initial_shape(self):
        return None

    @property
    def decoder_is_fixed(self):
        """Whether the decoder is fixed or learned - if true decoder matrix needs to return the matrix used to decode"""
        return False

    @property
    def decoder_matrix(self):
        """Matrix to use as decoder - only relevant if decoder_is_fixed"""
        return None

    """Likelihood training"""

    @property
    def use_likelihood(self):
        """If this is true the gaussian likelihood is maximized instead of using RMSE/BCE.
         (Obviously the likelihood is maximized by minimizing the negative log likelihood)"""
        return self._loss_fn == "likelihood"
    @property
    def individual_sigma(self):
        """If true learn one variance value for each entry of the output, if flase learns a mutual value for all entries,
        should be true if outputs are images"""
        return False

    @property
    def variance_decoder_dict(self):
        """Dictionary containing the config of the variance decoder hidden layer. This is followed by an output layer
         with tf.nn.elu + 1 activation, to ensure positive values """
        return self.encoder_dense_dict
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
        return self._batch_size

    @property
    def bptt_length(self):
        return self._bptt_length
    @property
    def scale_targets(self):
        """If OutputMode = RKN.OUTPUT_MODE_OBSERVATIONS then the targets are divided by scale_targets before
        computing the BCE.
        This is useful if you work with images - they can be divided by 255 to map them to [0-1] in order to compute
        tue bce. (This saves a lot of memory since you can keep them in uint8 and only transform the currently needed
        batch - the actual transformation to float32 happens implicitly when feeding the data into the model)
        """
        return None  # default = no rescaling (equivalent to return 1)

