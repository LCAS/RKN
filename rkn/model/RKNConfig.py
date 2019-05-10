import abc
from model.RKN import RKN
from transition_cell.TransitionCell import TransitionCell
import tensorflow as tf
class RKNConfig(abc.ABC):
    """Abstract class containing config for the RKN model - implement this for your own application"""


    def __init__(self, name):
        self.name = name
    '''Main Model Parameters'''

    @property
    def input_dim(self):
        """Dimensionality of the input data - currently only images supported
        give a list with [height, width, color channels]"""
        raise NotImplementedError("Input dim not given")

    @property
    def output_dim(self):
        """Dimensionality of the output data - either a scalar if RNK.OUTPUT_MODE_POSITIONS
         or a list with [height, width, color channels] if RKN.OUTPUT_MODE_OBSERVATIONS"""
        raise NotImplementedError("Output Dim not given")

    @property
    def output_mode(self):
        """Output mode - either RKN.OUTPUT_MODE_OBSERVATIONS or RKN.OUTPUT_MODE_POSITIONS"""
        raise NotImplementedError("Output Mode not given")

    @property
    def latent_observation_dim(self):
        """Dimensionality of the latent observations (In Paper: m)"""
        raise NotImplementedError("Latent Observation Dim not given")

    @property
    def latent_state_dim(self):
        """Dimensionality of the latent state (In Paper: n), implementation only supports
        latent_state_dim = 2 * self.latent_observation_dim"""
        return 2 * self.latent_observation_dim

    @property
    def variance_activation_fn(self):
        """'Activation' Function used to ensure all learned variances are positive"""
        raise NotImplementedError("Variance Activation Fn not given")

    @property
    def never_invalid(self):
        """If true it is assumed that all observations are valid and the update is not masked. This has no influence
        on the results but increases performance a bit"""
        return False

    @property
    def transition_cell_type(self):
        """Which Transition Cell to use:
        """

        return TransitionCell.TRANSITION_CELL_CORRELATED

    @property
    def n_step_pred(self):
        return -1

    @property
    def with_velocity_targets(self):
        return False

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
    def encoder_conv_dict(self):
        """Dictionary containing the config of the convolutional part of the encoder -
        see NConvHiddenLayers for details"""
        if self.only_dense_encoder:
            return None
        else:
            raise NotImplementedError("Encoder conv dict not given")

    @property
    def encoder_dense_dict(self):
        """Dictionary containing the config of the dense part of the encdder -
        see NDenseHiddenLayers for details"""
        raise NotImplementedError("Encoder dense dict not given")

    @property
    def only_dense_encoder(self):
        """If true the encoder_conv_dict is ignored and only a dense encoder (Based on encoder_dense_dict) is build"""
        return False

    @property
    def max_encoder_variance(self):
        """Maximum variance the encoder can output, everything above is clipped. If a negative value is returned
        clipping is disabled"""
        raise NotImplementedError("Max Encoder Variacne not given")

    @property
    def single_encoder_variance(self):
        """If true a single (shared) covariance for all latent observation entries is emitted, else individual ones"""
        raise NotImplementedError("Single Encoder Variance not given")

    @property
    def use_constant_observation_covariance(self):
        """Whether to learn a constant or input dependend variance"""
        return False

    @property
    def constant_observation_covariance_given(self):
        """If constant covariance is used, whether it is given or learned"""
        if self.use_constant_observation_covariance:
            raise NotImplementedError("Constant_observation_covariance_given not given")
        else:
            return None

    @property
    def constant_observation_covariance(self):
        """The constant observation covariance, if its given"""
        if self.constant_observation_covariance_given:
            raise NotImplementedError("constant_observation_covariance_given is true but constant_observation_covariance not given")
        return None

    @property
    def constant_observation_covariance_init(self):
        """Value to initialize the constant covariance with if its not given"""
        if self.constant_observation_covariance_given:
            raise NotImplementedError("Constant_observation_covariance_give is false but constant_observation_covariance_init not given")
        return None

    '''TransitionModel'''
    @property
    def bandwidth(self):
        """Bandwidth of the transition model"""
        raise NotImplementedError("Bandwidth not given")

    @property
    def init_mode(self):
        """Either INIT_MODE_COPY_OBSERVATION (observation part copied, memory part zeros)
        or INIT_MODE_RANDOM (all entries random)"""
        raise NotImplementedError("Initial mode not given")

    @property
    def transition_matrix_is_given(self):
        """Whether the transition matrix is given - if true transition_matrix needs to return the (true) transition
          matrix"""
        return False

    @property
    def transition_matrix(self):
        """The (true) transition matrix - only relevant if transition_matrix_given"""
        if self.transition_matrix_is_given:
            raise NotImplementedError("Transition matrix not given but transition_matrix_is_given true")

    @property
    def transition_matrix_mode(self):
        """What kind of transition matrix to use and how to initialize it"""
        raise NotImplementedError("Transition matrix mode not given")

    @property
    def transition_matrix_init(self):
        raise NotImplementedError("Transition matrix init not given")

    @property
    def transition_covariance_given(self):
        """Whether the transition noise is given - if true transition_noise_covar needs to return the (true) transition
        noise covariance"""
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
    def transition_covar_hidden_dict(self):
        if self.learn_state_dependent_transition_covar:
            raise NotImplementedError("Transition covar hidden dict not given but learn state dependent transition covar true")

    @property
    def transition_covariance(self):
        """True transition noise covariance in case its given"""
        if self.transition_covariance_given:
            raise NotImplementedError("Transition noise covar is not given but transition_noise_given true")

    @property
    def transition_covariance_init(self):
        """Value to initialize the transition covariance in case its learned"""
        if not self.transition_covariance_given:
            raise NotImplementedError("transition_noise_init not given but transition_noise_given false")

    @property
    def initial_state_covariance_given(self):
        """Whether a value for the initial state covariance is given"""
        return False

    @property
    def initial_state_covariance(self):
        """Initial state covariance to use if given"""
        if self.initial_state_covariance_given:
            raise NotImplementedError("initial_state_covariance not given but initial_state_covariance_given is true")

    @property
    def initial_state_covariance_init(self):
        """Value to initialize the initial state covariance in case its learned"""
        if not self.initial_state_covariance_given:
            raise NotImplementedError("initial_state_covariance_init not given but initial_state_covariance_given is false")

    @property
    def normalize_latent(self):
        """Whether to normalize latent_states and observations, this should be true except when working with known
        models and/or true spaces"""
        return True
    @property
    def normalize_obs(self):
        """Whether to normalize latent observations, this should be true except when working with known
        models and/or true spaces"""
        return True

    @property
    def normalize_prior(self):
        """Whether to normalize latent priors, this should be true except when working with known
        models and/or true spaces"""
        return True

    @property
    def normalize_posterior(self):
        """Whether to normalize the latent posterior, this should be true except when working with known
        models and/or true spaces"""
        return True



    @property
    def adapt_variance_to_normalization(self):
        if self.normalize_latent:
            raise NotImplementedError("not implemented")

    @property
    def use_sigmoid_in_normalization(self):
        return False

    '''Decoder'''
    @property
    def decoder_dense_position_dict(self):
        """Dictionary containing the config of the decoder if RKN in output mode is
         RKN.OUTPUT_MODE_POSITIONS"""
        raise NotImplementedError("Decoder dense position dict not given")

    @property
    def decoder_dense_observation_dict(self):
        """Dictionary containing the config of the dense part of the decoder if RKN in output mode is
         RKN.OUTPUT_MODE_OBSERVATIONS"""
        raise NotImplementedError("Decoder dense observation dict not given")
    
    @property
    def decoder_conv_dict(self):
        """Dictionary containing the config of the convolutional part of the decoder (only needed if output
        mode is RKN.OUTPUT_MODE_OBSERVATIONS)"""
        raise NotImplementedError("Decoder conv dict not given")

    @property
    def decoder_initial_shape(self):
        """List/Tuple containing the shape of the input to the first convolutional layer of the decoder (only needed if
        output mode is RKN.OUTPUT_MODE_OBSERVATIONS)"""
        raise NotImplementedError("Decoder Initial Shape not given")

    @property
    def decoder_mode(self):
        raise NotImplementedError("Not Implemented")

    @property
    def decoder_matrix(self):
        """Matrix to use as decoder - only relevant if decoder_mode == fixed"""
        return None

    """Likelihood training"""
    @property
    def use_likelihood(self):
        """If this is true the gaussian likelihood is maximized instead of using RMSE/BCE.
         (Obviously the likelihood is maximized by minimizing the negative log likelihood)"""
        return False

    @property
    def fix_likelihood_decoder(self):
        """If true the variance of the output is computed as the mean of the latent state variance, only works if
        individual_sigma is false"""
        if self.use_likelihood:
            raise NotImplementedError("Likelihood is used - but 'fix_likelihood_decoder' not given")

    @property
    def individual_sigma(self):
        """If true learn one variance value for each entry of the output, if false learns a mutual value for all entries,
        should be true if outputs are images"""
        if self.use_likelihood:
            raise NotImplementedError("Likelihood is used - but 'individual_sigma' not given")
        else:
            return None

    @property
    def variance_decoder_dict(self):
        """Dictionary containing the config of the variance decoder hidden layer. This is followed by an output layer
         with variance_activation_fn, to ensure positive values """
        if self.use_likelihood:
            raise NotImplementedError("Likelihhod is used - but 'variance_decoder_dict not given")
        else:
            return None


    '''Training'''
    @property
    def learning_rate(self):
        """Learning rate of the Adam Optimizer (\alpha)"""
        raise NotImplementedError("Learning rate not given")

    @property
    def reg_loss_factor(self):
        """Scaling factor of the L2 regularization loss"""
        raise NotImplementedError("L2 Reg Factor not given")

    @property
    def batch_size(self):
        """Batch size during training"""
        raise NotImplementedError("Batch Size not given")

    @property
    def bptt_length(self):
        """Steps before gradients are truncated for truncated Backprop Trough Time"""
        return NotImplementedError("tBPTT Length not given")

    @property
    def scale_targets(self):
        """If OutputMode = RKN.OUTPUT_MODE_OBSERVATIONS then the targets are divided by scale_targets before
        computing the BCE.
        This is useful if you work with images - they can be divided by 255 to map them to [0,1] in order to compute
        the bce. (This saves a lot of memory since you can keep them in uint8 and only transform the currently needed
        batch - the actual transformation to float32 happens implicitly when feeding the data into the model)
        """
        return None #default = no rescaling (equivalent to return 1)

    def mprint(self, *args, end=None):
        """Prints: <ModelName>: <s>, should be used instead of print for logging
        @:param s: string to print (as suffix)"""
        print(self.name + ":", *args, end=end)