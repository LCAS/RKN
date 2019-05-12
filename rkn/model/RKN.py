import tensorflow as tf
import numpy as np
from network import FeedForwardNet
from network import NConvolutionalHiddenLayers, NDenseHiddenLayers, HiddenLayerWrapper, ReshapeLayer, HiddenLayersParamsKeys
from network import SimpleOutputLayer, GaussianOutputLayer, UpConvOutputLayer, FixedOutputLayer
from network.LinearDecoder import LinearDecoder, FixedLinearDecoder
from util import MeanCovarPacker as mcp
from transition_cell import TransitionCell, RKNCorrTransitionCell
from util import GPUUtil as gpu_util
from util import LossFunctions as loss_fn


class RKN:
    """RKN Model: Builds the graph for the RKN Model"""

    """
    The model can be used to solve either of 2 Tasks:
    1) Filtering
    """
    TASK_MODE_FILTER = 'filter'
    """
    2)Prediction
    """
    TASK_MODE_PREDICT = 'predict'

    """
    The mode can produce either of 2 Outputs:
    1) Observations (reconstruct the (noise free) images) - This uses a transposed convolutional decoder
    """
    OUTPUT_MODE_OBSERVATIONS = 'observations'
    """
    2) Positions (or other low dimensional observations) - This uses a dense decoder
    """
    OUTPUT_MODE_POSITIONS = 'positions'

    """One of several possible transition matices can be used"""
    """N - dimensional spring damper (bandwidth = 0) initialized in a more or less stable manner"""
    TRANS_MATRIX_SPRING_DAMPER= 'sd'
    """N - dimensional spring damper (bandwdith = 0) - guaranteed by construction to be and stay stable"""
    TRANS_MATRIX_STABLE_SPRING_DAMPER = 'stable_sd'
    """Band matrix, potentially highly unstable, needs normalized latent space, initialized by identity + noise    """
    TRANS_MATRIX_BAND = 'band_matrix'
    """Band matrix 'clever' initialized to be more or less stable in the beginning, works without normalization.
    For 'clever' initialization each matrix is initialized such that the values in each column (and row) decrease 
    linearly towards the edges of the band and sum to 1. (potentially a little noise is added). The lower left and upper right
    band matrices are then multiplied with 0.1 and -0.1 respectively to mimic the behavior of the more or less stable spring damper"""
    #TRANS_MATRIX_BAND_SPRING_DAMPER = 'band_spring_damper'
    TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH = 'band_spring_damper_linear'
    TRANS_MATRIX_BAND_SPRING_DAMPER_STEP = 'band_spring_damper_step'
    TRANS_MATRIX_BAND_BOUNDED = 'band_spring_damper_bounded'
    TRANS_MATRIX_BAND_UNNORMAL = 'band_unnormal'

    """One of several possible decodes can be used"""
    """Fixed Matrix (usually if you work with linear dynamics and nonlinear observations, or for the LLRKN)"""
    DECODER_MODE_FIX = "decoder_fix"
    """Linear decoder: affine transformation of latent space posterior (https://math.stackexchange.com/a/332722) """
    DECODER_MODE_LINEAR = "decoder_lin"
    """Nonlinear decoder: full neural network, as specified in config"""
    DECODER_MODE_NONLINEAR = "decoder_nonlin"

    """One of two different ways to initialize the transition model, fix and random may have different meanings for 
    different transition matrix types"""
    TRANS_INIT_FIX = "trans_init_fix"
    TRANS_INIT_RAND = "trans_init_random"

    """ How to initialize the filter """
    """
    1) Initialize with a constant mean and a learned variance
    """
    INIT_MODE_CONST_MEAN_LEARN_VARIANCE = 'init_const_mean_learn_variance'

    def __init__(self, config, debug_recurrent=False):
        """
        Constructs new model
        :param config: RKNConfig Object containing the configuration data for the model
        :param debug_recurrent: If true the transition cell outputs not only the latent posterior distribution but also
                                the prior and kalman gain. Those are ignored by this class but can be fetched for visualization purposes
        """
        self.c = config
       
        self.c.mprint('Start building model ...')
        self.c.mprint('tensorflow version ', tf.__version__)
        self.output_mode = self.c.output_mode
        self.debug_recurrent = debug_recurrent
        self.latent_observation_dim = self.c.latent_observation_dim
        self.latent_state_dim = self.c.latent_state_dim
        assert 2 * self.latent_observation_dim == self.latent_state_dim, \
            "Latent state dimension needs to be two time latent observation dimension"


        self.transition_cell_type = self.c.transition_cell_type

        num_gpus = gpu_util.get_num_gpus()
        self.c.mprint('Found ', num_gpus, ' GPUs')
        self.use_gpu = False #num_gpus > 0 and not self.c.only_dense_encoder

        if self.use_gpu:
            self.input_dim = gpu_util.adapt_shape_for_gpu(self.c.input_dim)
            if self.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
                self.output_dim = gpu_util.adapt_shape_for_gpu(self.c.output_dim)
            else:
                self.output_dim = self.c.output_dim
        else:
            self.input_dim = self.c.input_dim
            self.output_dim = self.c.output_dim

        # ensures exception is thrown if model not fully executable on gpu
        if num_gpus > 0:
            with tf.device('/gpu:0'):
                self._build()
        else:
            self._build()
    """
    @property
    def debug_vars_labels(self):
        return ["norm", "tc_u", "tc_l", "ov_min", "ov_max", "ov_mean", "pr_min", "pr_max", "pr_mean", "po_min", "po_max", "po_mean"]

    @property
    def debug_vars(self):
        return [self.norm, tf.reduce_min(self.transition_covar_upper), tf.reduce_min(self.transition_covar_lower),
                tf.reduce_min(self.v_predictions), tf.reduce_max(self.v_predictions), tf.reduce_mean(self.v_predictions),
                tf.reduce_min(self.prior_covar[..., :self.latent_state_dim]),
                tf.reduce_max(self.prior_covar[..., :self.latent_state_dim]),
                tf.reduce_mean(self.prior_covar[..., :self.latent_state_dim]),
                tf.reduce_min(self.post_covar[..., :self.latent_state_dim]),
                tf.reduce_max(self.post_covar[..., :self.latent_state_dim]),
                tf.reduce_mean(self.post_covar[..., :self.latent_state_dim])]
    """
    def _build(self):
        """build individual parts of model"""
        self._build_inputs()
        self._build_encoder()
        self._build_transition_matrix()
        self._build_transition_noise_covar()
        self._build_initial_covar()
        self._build_transition_cell()
        self._build_initial_state()
        self._build_decoder()
        if self.c.use_likelihood:
            self._build_variance_decoder()
        self._build_model()

        self._build_loss()
        self._build_optimizer()
        
        self.c.mprint('... model successfully build')

    def _build_inputs(self):
        """ Creates input placeholders for model """
        with tf.name_scope("Inputs"):

            input_dim = self.input_dim if isinstance(self.input_dim, list) else [self.input_dim]
            self.observations = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None] + input_dim,
                                               name="observations")

            output_dim = self.output_dim if isinstance(self.output_dim, list) else [self.output_dim]
            self.targets = tf.placeholder(dtype=tf.float32,
                                          shape=[None, None] + output_dim,
                                          name='targets')

            self.observations_valid = tf.placeholder(dtype=tf.bool,
                                                     shape=[None, None, 1],
                                                     name="observations_valid")

            self.training = tf.placeholder(dtype=tf.bool,
                                           shape=[],
                                           name="training")

    def _build_initial_state(self):
        """builds initial state, if it is given just feed into initial_latent_state"""
        with tf.name_scope("InitialState"):
            batch_size = tf.shape(self.observations)[0]
            if self.c.init_mode == RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE:
                self.c.mprint("Initial State: Constant Mean, Learned Variance")
                if self.c.normalize_latent:
                    init_mean = tf.ones(shape=[batch_size, self.latent_state_dim])
                else:
                    init_mean = tf.zeros(shape=[batch_size, self.latent_state_dim])

                tile_suffix = [1, 1] if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL else [1]

                init_covar_ext = tf.tile(tf.expand_dims(self._initial_state_covar, 0), [batch_size] + tile_suffix)

                self.initial_latent_state = self.transition_cell.pack_state(init_mean, init_covar_ext)
            else:
                raise AssertionError("Currently only RKN.INIT_MODE_CONST_MEAN_LEARN_VARIANCE implemented")

    def _build_model(self):
        """ Plugs individual parts of the model together """
        with tf.name_scope("Model"):
            with tf.name_scope("Preprocessor"):
                if self.c.preprocessor is not None:
                    with tf.variable_scope('Preprocessor'):
                        self.encoder_input = self.c.preprocessor(self.observations, self.use_gpu)
                else:
                    self.encoder_input = self.observations
            with tf.name_scope("Encoder"):
                self.latent_observations = self.encoder(self.encoder_input, self.training, sequence_data=True)
                self.latent_observation_mean, self.latent_observations_covar \
                    = mcp.unpack(self.latent_observations, self.latent_observation_dim)
        #        self.latent_observations = tf.Print(self.latent_observations, [self.latent_observations],
        #                                            message="latent obs ", summarize=100)
            with tf.name_scope("PrepareInputs"):
                latent_obs_shape = tf.shape(self.latent_observations)
                padding_legnth = tf.shape(self.observations_valid)[1] - latent_obs_shape[1]
                padding = tf.ones((latent_obs_shape[0], padding_legnth, self.latent_observation_dim))
                padded_mean, padded_covar = [tf.concat([x, padding], 1) for x in [self.latent_observation_mean, self.latent_observations_covar]]

                if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
                    latent_obs_covar = tf.matrix_diag(padded_covar)
                else:
                    latent_obs_covar = padded_covar
                transition_model_inputs = \
                    self.transition_cell.pack_input(padded_mean, latent_obs_covar, self.observations_valid)

            with tf.name_scope("TransitionModel"):
                tm_output, self.last_state = \
                    tf.nn.dynamic_rnn(cell=self.transition_cell,
                                      inputs=transition_model_inputs,
                                      initial_state=self.initial_latent_state)

            with tf.name_scope("Decoder"):
                full_post_state = tm_output[:, :, :(5 * self.latent_observation_dim)]
                full_prior_state = tm_output[:, :, (5 * self.latent_observation_dim): (10 * self.latent_observation_dim)]
                if self.debug_recurrent:
                    upper = 2 * self.transition_cell.state_size + 3 * self.latent_observation_dim
                    self.transition_covar = tm_output[:, :, 2 * self.transition_cell.state_size :  upper ]
                    self.kalman_gain = tm_output[:, :, upper: ]
                self.prior_mean, self.prior_covar = self.transition_cell.unpack_state(full_prior_state)
                self.post_mean, self.post_covar = self.transition_cell.unpack_state(full_post_state)

                #self.prior_mean = tf.Print(self.prior_mean, [self.prior_mean], message="Prior Mean", summarize=50)
                #self.prior_covar = tf.Print(self.prior_covar, [self.prior_covar], message="Prior Covar", summarize=50)
                #self.post_mean = tf.Print(self.post_mean, [self.post_mean], message="Post Mean", summarize=50)
                #self.post_covar = tf.Print(self.post_covar, [self.post_covar], message="Post Covar", summarize=50)

                #with tf.control_dependencies([tf.print("prior mean", tf.norm(self.prior_mean)),
                #                              tf.print("prior covar", tf.norm(self.prior_covar)),
                #                              tf.print("post mean", tf.norm(self.post_mean)),
                #                              tf.print("post covar", tf.norm(self.post_covar))]):
                self.predictions = self.decoder(self.post_mean, self.training, sequence_data=True)

                if self.c.use_likelihood:
                    self.v_predictions = self.v_decoder(self.post_covar, self.training, sequence_data=True)

    def _build_transition_matrix(self):

        with tf.name_scope("TransitionModel"):
            if self.c.transition_matrix_init == RKN.TRANS_INIT_FIX:

                tm_11_init =         tf.eye(self.latent_observation_dim)
                tm_12_init =  0.05 * tf.eye(self.latent_observation_dim)
                tm_21_init = -0.05 * tf.eye(self.latent_observation_dim)
                tm_22_init =  0.95 * tf.eye(self.latent_observation_dim)
                tm_11_full = tf.get_variable(name="tm_11_basis", initializer=tm_11_init)
                tm_12_full = tf.get_variable(name="tm_12_basis", initializer=tm_12_init)
                tm_21_full = tf.get_variable(name="tm_21_basis", initializer=tm_21_init)
                tm_22_full = tf.get_variable(name="tm_22_basis", initializer=tm_22_init)
                tm_11, tm_12, tm_21, tm_22 = (tf.matrix_band_part(x, self.c.bandwidth, self.c.bandwidth) for x in
                                              [tm_11_full, tm_12_full, tm_21_full, tm_22_full])
                self.transition_matrix = tf.concat([tf.concat([tm_11, tm_12], -1),
                                                    tf.concat([tm_21, tm_22], -1)], -2)
            else:
                raise AssertionError("init mode not implemented")

        """
        def sample_and_check():
            sample = np.random.uniform(low=-0.1, high=0.1, size=[sub_matrix_dim, sub_matrix_dim])
            assert np.all(sample != 0.0), "Invalid transition matrix sampled"
            return sample
        with tf.name_scope("TransitionMatrix"):
            sub_matrix_dim = self.latent_observation_dim
            if self.c.transition_matrix_is_given:
                self.transition_matrix = tf.constant(self.c.transition_matrix, dtype=tf.float32)
            else:
                if self.c.transition_matrix_mode == RKN.TRANS_MATRIX_STABLE_SPRING_DAMPER:
                    self.c.mprint("Transition model: Stable Spring damper")
                    assert not self.c.normalize_latent, "Latent should not be normalized if stable spring damper transition model is used"
                    if self.c.transition_matrix_init == RKN.TRANS_INIT_FIX:
                        b = 0.1 * tf.ones(self.latent_observation_dim)
                    elif self.c.transition_matrix_init == RKN.TRANS_INIT_RAND:
                        b_init = -2.1972 * tf.ones(self.latent_observation_dim) # b = 0.1
                        b_learn = tf.get_variable(name='b_untransformed',
                                              initializer=b_init)
                        b = tf.nn.sigmoid(b_learn)

                    d_init = 4.369 * tf.ones(self.latent_observation_dim) # d = 0.95
                    d_learn = tf.get_variable(name='d_untransformed',
                                              initializer=d_init)
                    d = 4.0 * tf.nn.sigmoid(d_learn) - 3.0
                    #d = tf.Print(d, [d], "d")

                    val = 1.0 / b
                    c_lower = val * d - val
                    c_upper = tf.minimum(0.0, 2 * val * d + 2 * val)

                    c_init = tf.zeros(self.latent_observation_dim) #1.38629 * tf.ones(self.latent_observation_dim) # c = -0.01
                    c_learn = tf.get_variable(name='c_untransformed',
                                              initializer=c_init)
                    c_factor = tf.nn.sigmoid(c_learn)

                    c = c_factor * c_upper + (1 - c_factor) * c_lower
                    #c = tf.Print(c, [c], "c")

                    tm11 = tf.eye(sub_matrix_dim)
                    tm12 = tf.matrix_diag(b)
                    tm21 = tf.matrix_diag(c)
                    tm22 = tf.matrix_diag(d)
                elif self.c.transition_matrix_mode == RKN.TRANS_MATRIX_SPRING_DAMPER:
                    self.c.mprint("Transition model: Spring damper")
                    assert not self.c.normalize_latent, "Latent should not be normalized if spring damper transition model is used"
                    if self.c.transition_matrix_init == RKN.TRANS_INIT_RAND:
                        self.tm12_learn = tf.get_variable(name='tm11_learn', initializer=tf.random_uniform([sub_matrix_dim], minval=0.01, maxval=0.15))
                        self.tm21_learn = tf.get_variable(name='tm21_learn', initializer=tf.random_uniform([sub_matrix_dim], minval=-0.15, maxval=-0.01))
                    elif self.c.transition_matrix_init == RKN.TRANS_INIT_FIX:
                        self.tm12_learn = tf.get_variable(name='tm11_learn', initializer=0.1 * tf.ones(sub_matrix_dim))
                        self.tm21_learn = tf.get_variable(name='tm21_learn', initializer=-0.1 * tf.ones(sub_matrix_dim))
                    self.tm22_learn = tf.get_variable(name="tm22_learn", initializer=tf.ones(sub_matrix_dim))

                    tm11 = tf.eye(sub_matrix_dim)
                    tm12 = tf.matrix_diag(self.tm12_learn)
                    tm21 = tf.matrix_diag(self.tm21_learn)
                    tm22 = tf.matrix_diag(self.tm22_learn)

                elif self.c.transition_matrix_mode == RKN.TRANS_MATRIX_BAND:
                    self.c.mprint("Using Full diagonal transition model")
                  #  assert self.c.normalize_latent, "Latent should be normalized if band matrix is used"

                    tm11_init = sample_and_check()
                    np.fill_diagonal(tm11_init, 1.0)
                    tm11_full = tf.get_variable(name='tm11_full', initializer=tf.constant(tm11_init, dtype=tf.float32))
                    tm21_init = sample_and_check()
                    tm21_full = tf.get_variable(name='tm21_full', initializer=tf.constant(tm21_init, dtype=tf.float32))
                    tm12_init = sample_and_check()
                    tm12_full = tf.get_variable(name='tm12_full', initializer=tf.constant(tm12_init, dtype=tf.float32))
                    tm22_init = sample_and_check()
                    np.fill_diagonal(tm22_init, 1.0)
                    tm22_full = tf.get_variable(name='tm22_full', initializer=tf.constant(tm22_init, dtype=tf.float32))
                    tm11, tm21, tm12, tm22 = (tf.matrix_band_part(x, self.c.bandwidth, self.c.bandwidth) for x in
                                              [tm11_full, tm21_full, tm12_full, tm22_full])

                elif  "band_spring_damper" in self.c.transition_matrix_mode:
                    self.c.mprint("Using band spring damper transition model")
                #    assert not self.c.normalize_latent, "Latent should not be normalized if band spring damper transition model is used"
                    if self.c.transition_matrix_mode == RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH:
                        self.c.mprint("Transition Matrix Band SD: smooth")
                        def get_init_mat(off_diag):
                            off_fact = 0.1  / (2 * self.c.bandwidth) if self.c.bandwidth > 0 else 0
                            off_diag_fact = 0.1 / (2 * self.c.bandwidth + 1)
                            mat = np.zeros([sub_matrix_dim, sub_matrix_dim])
                            for i in range(-self.c.bandwidth, self.c.bandwidth + 1):
                                if off_diag:
                                    fact = off_diag_fact
                                else:
                                    fact = 0.9 if i == 0 else off_fact
                                curr = np.eye(sub_matrix_dim, k=i)
                                if self.c.transition_matrix_init == RKN.TRANS_INIT_RAND:
                                    curr += np.eye(sub_matrix_dim, k=i) * np.random.uniform(-0.1, 0.1, [sub_matrix_dim, sub_matrix_dim])
                                mat += fact * curr
                            return mat
                    elif self.c.transition_matrix_mode in [RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_STEP, RKN.TRANS_MATRIX_BAND_BOUNDED]:
                        self.c.mprint("Transition Matrix Band SD: step init")
                        def get_init_mat(off_diag):
                            mat = np.zeros([sub_matrix_dim, sub_matrix_dim])
                            off_fact = 0.1  / (2 * self.c.bandwidth)
                            for i in range(-self.c.bandwidth, self.c.bandwidth + 1):
                                fact = 0.9 if i == 0 else off_fact
                                curr = np.eye(sub_matrix_dim, k=i)
                                if self.c.transition_matrix_init == RKN.TRANS_INIT_RAND:
                                    curr += np.eye(sub_matrix_dim, k=i) * np.random.uniform(-0.1, 0.1, [sub_matrix_dim, sub_matrix_dim])
                                mat += fact * curr
                            return mat
                    else:
                        raise AssertionError("Invalid Transition Matrix Initialization")

                    if self.c.transition_matrix_mode == RKN.TRANS_MATRIX_BAND_BOUNDED:
                        self.c.mprint("enforce bound")
                        tm11_init = np.arctanh(get_init_mat(False))
                        tm12_init = np.arctanh(0.1 * get_init_mat(True))
                        tm21_init = np.arctanh(-0.1 * get_init_mat(True))
                        tm22_init = np.arctanh(get_init_mat(False))

                        tm11_full = tf.get_variable(name='tm11_full',
                                                    initializer=tf.constant(tm11_init, dtype=tf.float32))
                        tm21_full = tf.get_variable(name='tm21_full',
                                                    initializer=tf.constant(tm21_init, dtype=tf.float32))
                        tm12_full = tf.get_variable(name='tm12_full',
                                                    initializer=tf.constant(tm12_init, dtype=tf.float32))
                        tm22_full = tf.get_variable(name='tm22_full',
                                                    initializer=tf.constant(tm22_init, dtype=tf.float32))

                        tm11, tm21, tm12, tm22 = [tf.matrix_band_part(tf.tanh(x), self.c.bandwidth, self.c.bandwidth) for x in
                                                  [tm11_full, tm21_full, tm12_full, tm22_full]]

                    else:
                        self.c.mprint("Not Bounded")
                        tm11_init = get_init_mat(False)
                        tm12_init = 0.1 * get_init_mat(True)
                        tm21_init = -0.1 * get_init_mat(True)
                        tm22_init = get_init_mat(False)

                        tm11_full = tf.get_variable(name='tm11_full', initializer=tf.constant(tm11_init, dtype=tf.float32))
                        tm21_full = tf.get_variable(name='tm21_full', initializer=tf.constant(tm21_init, dtype=tf.float32))
                        tm12_full = tf.get_variable(name='tm12_full', initializer=tf.constant(tm12_init, dtype=tf.float32))
                        tm22_full = tf.get_variable(name='tm22_full', initializer=tf.constant(tm22_init, dtype=tf.float32))

                        tm11, tm21, tm12, tm22 = (tf.matrix_band_part(x, self.c.bandwidth, self.c.bandwidth) for x in
                                                 [tm11_full, tm21_full, tm12_full, tm22_full])

                elif self.c.transition_matrix_mode == RKN.TRANS_MATRIX_BAND_UNNORMAL:
                    self.c.mprint("Using unnormal band transition model")
                    tm11_init = sample_and_check() + 0.8 * np.eye(sub_matrix_dim)
                    tm11_full = tf.get_variable(name='tm11_full', initializer=tf.constant(tm11_init, dtype=tf.float32))
                    tm21_init = sample_and_check()
                    tm21_full = tf.get_variable(name='tm21_full', initializer=tf.constant(tm21_init, dtype=tf.float32))
                    tm12_init = sample_and_check()
                    tm12_full = tf.get_variable(name='tm12_full', initializer=tf.constant(tm12_init, dtype=tf.float32))
                    tm22_init = sample_and_check() + 0.8 * np.eye(sub_matrix_dim)
                    tm22_full = tf.get_variable(name='tm22_full', initializer=tf.constant(tm22_init, dtype=tf.float32))

                    tm11, tm21, tm12, tm22 = (tf.matrix_band_part(x, self.c.bandwidth, self.c.bandwidth) for x in
                                              [tm11_full, tm21_full, tm12_full, tm22_full])

                else:
                    raise AssertionError("Invalid Transition Matrix Initialization")
                self.transition_matrix = tf.concat([tf.concat([tm11, tm12], 1),
                                                    tf.concat([tm21, tm22], 1)], 0)
                #self.transition_matrix = tf.Print(self.transition_matrix, [self.transition_matrix], message="transitionMat", summarize=50)
        """

    def _build_transition_noise_covar(self):
        var_size = self.latent_observation_dim if self.c.inidividual_transition_covar else 1
        self.c.mprint("Using individual transition covariance:", self.c.inidividual_transition_covar)
        if self.c.learn_state_dependent_transition_covar:
            self._build_state_dependent_transition_covar(var_size)
        else:
            self._build_constant_transition_covar(var_size)

    def _build_state_dependent_transition_covar(self, var_size):
        self.c.mprint("Learning State Dependent Trainsition Covar")
        #covar_facts = tf.get_variable(name="covar_facts",
        #                              dtype=tf.float32,
        #                              shape=[1, self.latent_state_dim],
        #                              initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        #covar_offset = tf.get_variable(name="covar_offset",
        #                               initializer=tf.constant(np.log(self.c.transition_covariance_init[0]),
        #                                                       shape=[1, self.latent_state_dim],
        #                                                       dtype=tf.float32))
        covar_learner_out_dim = (3 if self.c.learn_correlated_transition_covar else 2) * var_size
        transition_covar_hidden_layer = NDenseHiddenLayers(self.c.transition_covar_hidden_dict, name_prefix="trans_covar_hidden")
        transition_covar_out_layer = SimpleOutputLayer(output_dim=covar_learner_out_dim, name_prefix="trans_covar_out")
        trans_covar_learner = FeedForwardNet(transition_covar_out_layer, transition_covar_hidden_layer)
        def _trans_covar_fn(state):
        #    trans_upper_lower = tf.exp(covar_facts * state + covar_offset)
        #    return self._construct_transition_covar([trans_upper_lower[..., :self.latent_observation_dim],
        #                                             trans_upper_lower[..., self.latent_observation_dim:],
        #                                             tf.zeros([tf.shape(trans_upper_lower)[0], var_size], dtype=tf.float32)])
            trans_covar_raw = trans_covar_learner(state)
            trans_covar_upper = self.c.variance_activation_fn(trans_covar_raw[..., :var_size])
            trans_covar_lower = self.c.variance_activation_fn(trans_covar_raw[..., 1 * var_size: 2 * var_size])
            covars = [trans_covar_upper, trans_covar_lower]
            if self.transition_cell_type in [TransitionCell.TRANSITION_CELL_CORRELATED, TransitionCell.TRANSITION_CELL_FULL]:
                if self.c.learn_correlated_transition_covar:
                    trans_covar_side_facts = tf.tanh(trans_covar_raw[..., 2 * var_size:])
                    trans_covar_side = trans_covar_side_facts * tf.minimum(trans_covar_upper, trans_covar_lower)
                else:
                    trans_covar_side = tf.zeros([tf.shape(trans_covar_raw)[0], var_size], dtype=tf.float32)
                covars += [trans_covar_side]
            return self._construct_transition_covar(covars)
        self._transition_covar = _trans_covar_fn

    def _build_constant_transition_covar(self, var_size):
        if self.c.transition_covariance_given:
            self.c.mprint("Using given transition covar")
            covars = [tf.constant(x, dtype=tf.float32, shape=[1, var_size]) for x in self.c.transition_covariance]
        else:
            self.c.mprint("Learning Constant Transition Covar")
            init_upper, init_lower = self.c.transition_covariance_init[:2]
            log_transition_covar_upper = tf.get_variable(name="log_transition_covar_upper",
                                                         dtype=tf.float32,
                                                         initializer=tf.constant(np.log(init_upper),
                                                                                 dtype=tf.float32,
                                                                                 shape=[1, var_size]))
            self.transition_covar_upper = self.c.variance_activation_fn(log_transition_covar_upper)

            log_transition_covar_lower = tf.get_variable(name="log_transition_covar_lower",
                                                         dtype=tf.float32,
                                                         initializer=tf.constant(np.log(init_lower),
                                                                                 dtype=tf.float32,
                                                                                 shape=[1, var_size]))
            self.transition_covar_lower = self.c.variance_activation_fn(log_transition_covar_lower)
            covars = [self.transition_covar_upper, self.transition_covar_lower]
            if self.transition_cell_type in [TransitionCell.TRANSITION_CELL_CORRELATED, TransitionCell.TRANSITION_CELL_FULL]:
                # In order for the transition noise covariance matrix to be positive definite the "side" entry always needs
                # to be smaller than both, the value for the upper and lower diagonal. (Strictly diagonal dominant,
                #  symmetric matrix is positive definite)
                if self.c.learn_correlated_transition_covar:
                    init_side = self.c.transition_covariance_init[2]
                    assert init_side < init_upper and init_side < init_lower
                    init_side_fact = init_side / np.minimum(init_upper, init_lower)
                    trans_covar_side_fact_arctanh = tf.get_variable(name="transition_covar_side_factor_logit",
                                                                    dtype=tf.float32,
                                                                    initializer=tf.constant(np.arctanh(init_side_fact),
                                                                                            dtype=tf.float32,
                                                                                            shape=[1, var_size]))
                    self.transition_covar_side = tf.tanh(trans_covar_side_fact_arctanh) * \
                                                 tf.minimum(self.transition_covar_upper, self.transition_covar_lower)
                else:
                    self.transition_covar_side = tf.constant(0.0, shape=[1, var_size], dtype=tf.float32)
                covars += [self.transition_covar_side]
    #                self.transition_covar_side = tf.Print(self.transition_covar_side, [self.transition_covar_side], message="Side")
        self._transition_covar = self._construct_transition_covar(covars)

    def _construct_transition_covar(self, covars):
        if self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
            covars = [tf.eye(self.latent_observation_dim) * tf.expand_dims(x, -1) for x in covars]
            return tf.concat([tf.concat([covars[0], covars[2]], -2),
                              tf.concat([covars[2], covars[1]], -2)], -1)
        elif not self.c.inidividual_transition_covar:
            return tf.concat([tf.tile(x, [1, self.latent_observation_dim]) for x in covars], -1)
        else:
            return tf.concat(covars, -1)

    def _build_initial_covar(self):
        if self.c.initial_state_covariance_given:
            self.initial_state_covar = tf.constant([self.c.initial_state_covariance], dtype=tf.float32)
        else:
            log_initial_state_covar = tf.get_variable(name="log_initial_state_covar",
                                                      initializer=tf.constant([np.log(self.c.initial_state_covariance_init)], dtype=tf.float32))
            self.initial_state_covar = self.c.variance_activation_fn(log_initial_state_covar)
        if self.transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
            self._initial_state_covar = tf.tile(self.initial_state_covar, [self.latent_state_dim])
        elif self.transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
            self._initial_state_covar = tf.concat([tf.tile(self.initial_state_covar, [self.latent_state_dim]), tf.zeros(self.latent_observation_dim)], 0)
        elif self.transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
            self._initial_state_covar = self.initial_state_covar * tf.eye(self.latent_state_dim)

    def _build_transition_cell(self):
        """ Builds transition model """
        if self.c.transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
            self.c.mprint("using Simple RKN Transition Cell")
            self.transition_cell = RKNSimpleTransitionCell(config=self.c,
                                                           transition_matrix=self.transition_matrix,
                                                           transition_covar=self._transition_covar,
                                                           debug=self.debug_recurrent)

        elif self.c.transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
            self.c.mprint("using Correlated RKN Transition Cell")
            self.transition_cell = RKNCorrTransitionCell(config=self.c,
                                                         transition_matrix=self.transition_matrix,
                                                         transition_covar=self._transition_covar,
                                                         debug=self.debug_recurrent)
        else:
            self.c.mprint("using full rkn transition cell")
            self.transition_cell = RKNFullTransitionCell(config=self.c,
                                                         transition_matrix=self.transition_matrix,
                                                         transition_covar=self._transition_covar,
                                                         debug=self.debug_recurrent)
            #raise AssertionError("Invalid transition cell type (Full Cell only for LLRKN feasable")

    def _build_encoder(self):
        """ Builds encoder network """
        encoder_hidden_dense = NDenseHiddenLayers(params=self.c.encoder_dense_dict,
                                                  name_prefix='EncoderHiddenDense')

        if self.c.only_dense_encoder:
            encoder_hidden = encoder_hidden_dense
        else:
            encoder_hidden_conv = NConvolutionalHiddenLayers(params=self.c.encoder_conv_dict,
                                                             name_prefix='EncoderHiddenConv',
                                                             on_gpu=self.use_gpu,
                                                             flatten_output=True,
                                                             up_convolutional=False)
            encoder_hidden = HiddenLayerWrapper([encoder_hidden_conv, encoder_hidden_dense])

        encoder_out = GaussianOutputLayer(distribution_dim=self.latent_observation_dim,
                                          name_prefix='EncoderOut',
                                          single_variance=self.c.single_encoder_variance,
                                          max_variance=self.c.max_encoder_variance,
                                          variance_fn=self.c.variance_activation_fn,
                                          constant_variance=self.c.use_constant_observation_covariance,
                                          fix_constant_variance=self.c.constant_observation_covariance_given,
                                          variance=self.c.constant_observation_covariance if self.c.constant_observation_covariance_given
                                          else self.c.constant_observation_covariance_init,
                                          normalize_mean=self.c.normalize_obs)

        self.encoder = FeedForwardNet(output_layer=encoder_out,
                                      hidden_layers=encoder_hidden)

    def _build_decoder(self):
        """ Builds decoder network """
        if self.c.decoder_mode == RKN.DECODER_MODE_FIX:
            self.c.mprint("Fixed Decoder")

            #decoder_hidden = None

            #decoder_out = FixedOutputLayer(output_dim=self.output_dim,
            #                               name_prefix='DecoderOut',
            #                               weight_matrix=self.c.decoder_matrix.T)
            self.lin_decoder = FixedLinearDecoder(w=tf.constant(self.c.decoder_matrix.T, dtype=tf.float32)) #FeedForwardNet(output_layer=decoder_out, hidden_layers=decoder_hidden)
            #self.lin_decoder = LinearDecoder(input_dim=self.latent_observation_dim, output_dim=self.output_dim)
            self.decoder = lambda x, training, sequence_data: \
                self.lin_decoder(x[:, :, :self.latent_observation_dim], training, sequence_data)

        elif self.c.decoder_mode == RKN.DECODER_MODE_NONLINEAR and self.output_mode == RKN.OUTPUT_MODE_POSITIONS:
            self.c.mprint("Nonlinear Decoder")
            if self.c.decoder_dense_position_dict is not None:
                decoder_hidden = NDenseHiddenLayers(params=self.c.decoder_dense_position_dict,
                                                    name_prefix='DecoderHidden')
            else:
                decoder_hidden = None
            decoder_out = SimpleOutputLayer(output_dim=self.output_dim,
                                            name_prefix='DecoderOut',
                                            activation=tf.identity)
            self.decoder = FeedForwardNet(output_layer=decoder_out, hidden_layers=decoder_hidden)


        elif self.c.decoder_mode == RKN.DECODER_MODE_NONLINEAR and self.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
            self.c.mprint("Nonlinear Decoder")
            decoder_hidden_dense = NDenseHiddenLayers(params=self.c.decoder_dense_observation_dict,
                                                      name_prefix='DecoderHiddenDense')
            if self.c.decoder_conv_dict is not None:
                decoder_hidden_reshape = ReshapeLayer(shape=self.c.decoder_initial_shape,
                                                      name_prefix="DecoderHiddenReshape",
                                                      on_gpu=self.use_gpu)
                decoder_hidden_conv = NConvolutionalHiddenLayers(params=self.c.decoder_conv_dict,
                                                                 name_prefix='DecoderHiddenUpConv',
                                                                 on_gpu=self.use_gpu,
                                                                 flatten_output=False,
                                                                 up_convolutional=True)
                decoder_hidden = HiddenLayerWrapper([decoder_hidden_dense,
                                                     decoder_hidden_reshape,
                                                     decoder_hidden_conv])
            else:
                decoder_hidden = decoder_hidden_dense
            decoder_out = UpConvOutputLayer(output_dim=self.output_dim,
                                            name_prefix='DecoderOut',
                                            on_gpu=self.use_gpu,
                                            activation=tf.nn.sigmoid)
            self.decoder = FeedForwardNet(output_layer=decoder_out, hidden_layers=decoder_hidden)

        elif self.c.decoder_mode == RKN.DECODER_MODE_LINEAR:
            self.c.mprint("Linear Decoder")
            dim = self.latent_state_dim if self.c.with_velocity_targets else self.latent_observation_dim
            self.lin_decoder = LinearDecoder(input_dim=dim, output_dim=self.output_dim)
            self.decoder = lambda x, training, sequence_data: \
                self.lin_decoder(x[:, :, :dim], training, sequence_data)
        else:
            raise AssertionError('Invalid decoder mode')

    def _build_variance_decoder(self):
        if self.c.decoder_mode in [RKN.DECODER_MODE_LINEAR, RKN.DECODER_MODE_FIX]:
            self.v_decoder = lambda x, _, sequence_data:\
                self.lin_decoder.decode_diagonal_variance(x[:, :, :self.latent_observation_dim], False, sequence_data)
        elif self.c.fix_likelihood_decoder:
            if self.c.individual_sigma:
                if self.output_dim == self.latent_observation_dim:
                    self.v_decoder = lambda x, training, sequence_data : x[:, :, :self.latent_observation_dim]
                else:
                    assert self.latent_observation_dim % self.output_dim == 0, "If post var should be split up the output dim needs to evenly divide latent observation dim"
                    def split_mean_decoader(x, in_dim, out_dim):
                        size = (int) (in_dim / out_dim)
                        parts = [x[:, :, i * size : (i + 1) * size] for i in range(out_dim)]
                        means = [tf.reduce_mean(x, axis=-1, keepdims=True) for x in parts]
                        return tf.concat(means, axis=-1)
                    self.v_decoder = lambda x, training, sequence_data: split_mean_decoader(x, self.latent_observation_dim, self.output_dim)
            else:
                self.v_decoder = lambda x, training, sequence_data: tf.reduce_mean(x[:, :, :self.latent_observation_dim], axis=-1, keepdims=True)
        else:
            if self.c.variance_decoder_dict is not None:
                v_decoder_hidden_dense = NDenseHiddenLayers(params=self.c.variance_decoder_dict,
                                                            name_prefix='DecoderVarianceHiddenDense')
            else:
                v_decoder_hidden_dense = None
            v_decoder_out = SimpleOutputLayer(output_dim=self.output_dim if self.c.individual_sigma else 1,
                                              name_prefix='DecoderVarianceOut',
                                              activation=self.c.variance_activation_fn)
            self.v_decoder = FeedForwardNet(output_layer=v_decoder_out,
                                            hidden_layers=v_decoder_hidden_dense)

    def _build_loss(self):
        """ Defines the loss function """
        # Prediction Loss
        with tf.name_scope("Loss"):
            if self.c.use_likelihood:
                reconstruction_loss_nll = loss_fn.gaussian_nll(targets=self.targets,
                                                               predictions=self.predictions,
                                                               variance=self.v_predictions,
                                                               img_data=self.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS,
                                                               scale_targets=self.c.scale_targets)

            if self.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
                self.reconstruction_loss = loss_fn.binary_crossentropy(targets=self.targets,
                                                                       predictions=self.predictions,
                                                                       img_data=True,
                                                                       scale_targets=self.c.scale_targets)
            elif self.output_mode == RKN.OUTPUT_MODE_POSITIONS:
                self.reconstruction_loss = loss_fn.rmse(targets=self.targets, predictions=self.predictions)
            else:
                raise AssertionError('Invalid output mode')
            # Latent Observation covariance L2 Regularization
            if self.c.reg_loss_factor > 0:
                prior_loss = (tf.reduce_sum(self.prior_mean**2, axis=-1) - 1) ** 2
                posterior_loss = (tf.reduce_sum(self.post_mean**2, axis=-1) - 1) ** 2
                regularization_loss = self.c.reg_loss_factor * (tf.reduce_mean(posterior_loss) + tf.reduce_mean(prior_loss))
               #latent_observation_covar = self.latent_observations[:, :, self.latent_observation_dim:]
               #point_wise_regularization_loss = tf.reduce_sum(latent_observation_covar**2, axis=-1)
               #regularization_loss = self.c.l2_reg_factor * tf.reduce_mean(point_wise_regularization_loss)
            else:
               regularization_loss = tf.zeros([])

            if self.c.use_likelihood:
                self.reference_loss = self.reconstruction_loss
                self.reconstruction_loss = reconstruction_loss_nll
            else:
                self.reference_loss = self.reconstruction_loss

            self.loss = [self.reconstruction_loss, regularization_loss, self.reference_loss]
            #self.loss = tf.Print(self.loss, [self.loss], message="loss")

    def _build_optimizer(self):
        """ Builds the optimizer nodes"""
        with tf.name_scope("Optimizer"):
            total_loss = tf.reduce_sum(self.loss[:2])
            opt = tf.train.AdamOptimizer(learning_rate=self.c.learning_rate)
            grad, vars = zip(*opt.compute_gradients(total_loss))
            grad, _ = tf.clip_by_global_norm(grad, 5.0)
            if np.isscalar(self.c.learning_rate):
                self.optimizer = opt.apply_gradients(zip(grad, vars))
            else:
                self.global_step = tf.get_variable("global_step", initializer=0.0, dtype=tf.float32)
                self.optimizer = opt.apply_gradients(zip(grad, vars), global_step=self.global_step)

