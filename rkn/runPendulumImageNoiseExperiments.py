import numpy as np
import tensorflow as tf

from data import Pendulum
from data.MultiplePendulumData import MultiplePendulums
from config.PendulumImageNoiseConfig import PendulumImageNoiseConfig
from model import RKN, RKNRunner
from baselines import LSTMBaseline, LSTMBaselineRunner, GRUBaseline, GRUBaselineRunner
from model_local_linear import LLRKN

from plotting import PendulumPlotter

""" For developing - overwrites flags!"""
# Todo remove before "release"
#DEBUG = False
##DEBUG_TASK_MODE = "filter"
#DEBUG_TASK_MODE = "filter"
#DEBUG_OBSERVATION_MODE = "positions"
##DEBUG_OBSERVATION_MODE = "observations"

#DEBUG_TRANSITION_MODEL = "rkn"

plotting = True
debug_rkn = False

""" EXPERIMENT CONFIGURATION"""
flags = tf.app.flags

flags.DEFINE_string("name", "model", "name of the model")
flags.DEFINE_string("output_mode", "obs", "either pos or obs")
flags.DEFINE_string("task_mode", "pred", "either pred or filt")
#flags.DEFINE_float("pendulum_friction", 0.1, "Friction factor of the pendulum")
#flags.DEFINE_string("task_mode", "filter", "task mode of the model, either 'filter' or 'predict'")
#flags.DEFINE_string("output_mode", "positions", "output mode of the model, either 'positions' or 'observations'")
flags.DEFINE_integer("latent_obs_dim", 250, "dimensionality of latent observations")
flags.DEFINE_integer("band_width", 15, "Band width of transition matrix")
flags.DEFINE_integer("num_basis", 15, "Number of basis matrices")
flags.DEFINE_string("model_type", "rknc", "which model to use, either 'rkns', 'rknc', 'llrkn' or 'lstm'")
#lags.DEFINE_string("trans_init", "fix", "fix or rand")
#flags.DEFINE_string("transition_matrix", "band_sd", "which type of transition matrix to use in case of rkn, either 'band', 'sd' (spring damper), or 'ssd' (stable spring damper)")
#flags.DEFINE_bool("img_noise", True, "whether to add noise to the images")
#flags.DEFINE_float("observation_noise_std", 0.001, "how much observation noise to add to the angular position of the pendulun")
flags.DEFINE_integer("training_intervals", 30, "number of intervals to train for")
flags.DEFINE_string("decoder_mode", "nonlin", "Which decoder to use, currently 'lin' and 'nonlin' supported")
#flags.DEFINE_integer("training_epochs", 500, "amount of training epochs")
#flags.DEFINE_integer("give_first_n", 5, "How many observations are given in case of prediction (ignored for filtering)")
#flags.DEFINE_integer("eval_iters", 10, "number of iterations the evaluation is repeated")
#flags.DEFINE_string("transition_model_type", "SpringDamper", "Either 'SpringDamper', 'Stable' or 'Band'")
flags.DEFINE_bool("use_ll", True, "Whether to use the negative log likelihood as cost function")
flags.DEFINE_bool("norm_latent", False, "")
flags.DEFINE_bool("multiple_pendulums", True, "")
#flags.DEFINE_bool("norm_post", False, "")
#flags.DEFINE_bool("norm_prior", False, "")
#flags.DEFINE_bool("adapt_var", False, "")
#flags.DEFINE_float("reg_loss_fact", 0, "")
#flags.DEFINE_float("tc_init_lower", 0.1, "")
#flags.DEFINE_float("tc_init_upper", 0.1, "")
#flags.DEFINE_integer("n_step_pred", 0, "")
#flags.DEFINE_float("sc_init", 1., "")



#flags.DEFINE_integer("band_width", 3, "Bandwidth of transition matrix")
#flags.DEFINE_bool("state_dep_trans_covar", False, "Whether to learn a state dependent transition covar")
#flags.DEFINE_bool("corr_trans_covar", False, "Whether to learn a correlated dependent transition covar")
#flags.DEFINE_bool("indi_trans_covar", False, "Whether to learn a individual value for each entry of the transition covar")
#flags.DEFINE_bool("fix_l_decoder", False, "Whether to compute the variance as mean of latent or by NN")
#flags.DEFINE_integer("train_episodes", 600, "Number of epochs to use for training")
flags.DEFINE_integer("seed", 0, "random seed for the data generator")
#flags.DEFINE_bool("sigmoid_before_norm", False, "Whether to apply sigmoid before normalization")

name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)

m_print("Starting Pendulum Experiment")
#if DEBUG:
#    m_print("!!!DEBUGING!!! - flags overwritten")

"""Configure Task Mode"""
#if DEBUG:
#    task_mode = DEBUG_TASK_MODE
#else:
#    if flags.FLAGS.task_mode == "filter":
#        m_print("Task Mode: Filter")
#        task_mode = RKN.TASK_MODE_FILTER
#    elif flags.FLAGS.task_mode == "predict":
#        m_print("Task Mode: Predict")
#        task_mode = RKN.TASK_MODE_PREDICT
#    else:
#        raise AssertionError("Invalid Task mode given, needs to be either 'filter' or 'predict'")

#if DEBUG:
#    output_mode = DEBUG_OBSERVATION_MODE
#else:
#    if flags.FLAGS.output_mode == "positions":
#        m_print("Output Mode: Positions")
#        output_mode = RKN.OUTPUT_MODE_POSITIONS
#    elif flags.FLAGS.output_mode == "observations":
#        m_print("Output Mode: Observations")
#        output_mode = RKN.OUTPUT_MODE_OBSERVATIONS
#    else:
#        raise AssertionError("Invalid Output Mode given, needs to be either 'positions' or 'observations'")




model_type = flags.FLAGS.model_type
m_print("Using", model_type, "as transition cell")

if flags.FLAGS.output_mode == "pos":
    output_mode = RKN.OUTPUT_MODE_POSITIONS
    m_print("Output Mode: Positions")
elif flags.FLAGS.output_mode == "obs":
    output_mode = RKN.OUTPUT_MODE_OBSERVATIONS
    m_print("Output Mode: Observations")
else:
    raise AssertionError("Invalid output mode")

if flags.FLAGS.task_mode == "filt":
    task_mode = RKN.TASK_MODE_FILTER
    m_print("Task Mode: Filter")
elif flags.FLAGS.task_mode == "pred":
    task_mode = RKN.TASK_MODE_PREDICT
    m_print("Task Mode: Prediction")
else:
    raise AssertionError("Invalid task mode")

transition_matrix = RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH
m_print("Using Smooth RKN Initialization")

try:
    seed_offset = int(name[-1]) - 1
except:
    seed_offset = 0
data_seed = flags.FLAGS.seed + seed_offset
m_print("Data Seed", data_seed)

multiple_pendulums = flags.FLAGS.multiple_pendulums
m_print("Multiple Pendulums:", multiple_pendulums)

pendulum_friction = 0.1 #flags.FLAGS.pendulum_friction
m_print("Pendulum Friction", pendulum_friction)

latent_observation_dim = flags.FLAGS.latent_obs_dim
m_print("Latent Observation Dim", latent_observation_dim)

if flags.FLAGS.decoder_mode == "nonlin":
    decoder_mode = RKN.DECODER_MODE_NONLINEAR
    m_print("Using nonlnear decoder")
elif flags.FLAGS.decoder_mode == "lin":
    decoder_mode = RKN.DECODER_MODE_LINEAR
    m_print("Using linear decoder")
else:
    raise AssertionError("Invalid Decoder Mode")

transition_matrix_init = RKN.TRANS_INIT_FIX
m_print("Fix initialization")

training_intervals = flags.FLAGS.training_intervals
m_print("Training for ", training_intervals, "intervals")

use_likelihood = flags.FLAGS.use_ll
m_print("Use Likelihood:", use_likelihood)

normalize_latent = flags.FLAGS.norm_latent
m_print("Normalize Latent:", normalize_latent)

fix_likelihood_decoder = False #model_type != "lstm"#flags.FLAGS.fix_l_decoder
m_print("Fix Likelihood Decoder: ", fix_likelihood_decoder)

train_episodes = 50 if task_mode == RKN.TASK_MODE_FILTER else 50 #flags.FLAGS.train_episodes
m_print("Train Episodes:", train_episodes)

observation_noise_std = 1e-5# flags.FLAGS.observation_noise_std
m_print("Observation noise standard deviation:", observation_noise_std)

tc_init_upper = 0.1 #flags.FLAGS.tc_init_upper
m_print("Transition Noise Covar Init Upper:", tc_init_upper)

tc_init_lower = 0.1#flags.FLAGS.tc_init_lower
m_print("Transition Noise Covar Init Lower:", tc_init_lower)

sc_init = 1.0 # flags.FLAGS.sc_init
m_print("State Covar Init:", sc_init)

img_noise = True #flags.FLAGS.img_noise
m_print("Using Image Noise:", img_noise)

num_basis = flags.FLAGS.num_basis
m_print("Num Basis", num_basis)

band_width = flags.FLAGS.band_width
m_print("Band Width:", band_width)

state_dep_trans_covar = True # flags.FLAGS.state_dep_trans_covar
m_print("Learning state dependent transition covar:", state_dep_trans_covar)

corr_trans_covar = False #flags.FLAGS.corr_trans_covar
m_print("Learning correlated transition covar:", corr_trans_covar)

indi_trans_covar = True #flags.FLAGS.indi_trans_covar
m_print("Learning individual values ofr state covariance entries:", indi_trans_covar)

sigmoid_before_normalization = False #flags.FLAGS.sigmoid_before_norm
m_print("Applying sigmoid before normalization:", sigmoid_before_normalization)

with_velo = False
""" Experiment Constants"""

img_size = [24, 24, 1]
episode_length = 150 if task_mode == RKN.TASK_MODE_FILTER else  150
test_episodes = 500 if task_mode == RKN.TASK_MODE_FILTER else 500


""" Build and Run Model"""

config = PendulumImageNoiseConfig(name=name,
                                  latent_observation_dim=latent_observation_dim,
                                  model_type=model_type,
                                  transition_matrix=transition_matrix,
                                  use_likelihood=use_likelihood,
                                  task_mode=task_mode,
                                  trans_matrix_init=transition_matrix_init,
                                  decoder_mode=decoder_mode,
                                  band_width=band_width,
                                  num_basis=num_basis,
                                  multiple_pendulums=multiple_pendulums,
                                  correlated_transition_covar=corr_trans_covar,
                                  individual_transition_covar=indi_trans_covar,
                                  output_mode=output_mode,
                                  trans_covar_init_lower=tc_init_lower,
                                  trans_covar_init_upper=tc_init_upper,
                                  state_covar_init=sc_init,
                                  normalize_latent=normalize_latent,
                                  learn_state_dependent_transition_covar=False,
                                  reg_loss_factor=0)

if model_type[:3] == 'rkn':
    model = RKN(config=config, debug_recurrent=task_mode==RKN.TASK_MODE_FILTER)
    model_runner = RKNRunner(model)
elif model_type == 'llrkn':
    model = LLRKN(config=config, debug_recurrent=False)
    #model = LLRKNFactorized(config=config, debug_recurrent=task_mode==RKN.TASK_MODE_FILTER)
  #  model = FFLLRKNBasis(config=config, debug_recurrent=task_mode==RKN.TASK_MODE_FILTER)
  #  model = LLRKNFactorizedBasis(config=config, debug_recurrent=False)
    model_runner = RKNRunner(model)
elif model_type == 'lstm':
    model = LSTMBaseline(config=config)
    model_runner = LSTMBaselineRunner(model)
elif model_type == 'gru':
    model = GRUBaseline(config=config)
    model_runner = GRUBaselineRunner(model=model)
else:
    raise AssertionError("Invalid tranistion Model")

#historgram_dict = {"encoder covar": model.latent_observations_covar}
#scalar_dict = {"reconstruction loss": model.reconstruction_loss,
#               "reference loss:": model.reference_loss}

"""Prepare Data"""
m_print("Start Generating Pendulum Data...")
pend_params = MultiplePendulums.pendulum_default_params()
pend_params[MultiplePendulums.FRICTION_KEY] = pendulum_friction

if multiple_pendulums:
    data = MultiplePendulums(24,
                             observation_mode=MultiplePendulums.OBSERVATION_MODE_LINE,
                             num_pends=3,
                             transition_noise_std=0.1,
                             observation_noise_std=observation_noise_std,
                             seed=data_seed,
                             pendulum_params=pend_params)
else:
    """CHANGE HERE!!!"""
    data = Pendulum(24,observation_mode=MultiplePendulums.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1, #HERE!
                    observation_noise_std=observation_noise_std,
                    seed=data_seed,
                    pendulum_params=pend_params)

def prepare_data(num_episodes, t):
    obs_raw, targets_raw, _, _ = data.sample_data_set(num_episodes, episode_length, with_velo, seed=1 if t else 0)
    if task_mode == RKN.TASK_MODE_FILTER:
        obs = obs_raw[:, :episode_length]
        obs, factors = data.add_observation_noise(obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
        if len(obs.shape) == 4:
            obs = np.expand_dims(obs, -1)
        if output_mode == RKN.OUTPUT_MODE_POSITIONS:
            targets = targets_raw
        else:
            targets = np.expand_dims(obs_raw, -1) if len(obs_raw.shape) == 4 else obs_raw
        obs_valid = None
    elif task_mode == RKN.TASK_MODE_PREDICT:
        obs = obs_raw[:, :episode_length]
        if len(obs.shape) == 4:
            obs = np.expand_dims(obs, -1)
        if output_mode == RKN.OUTPUT_MODE_POSITIONS:
            targets = targets_raw[:, :episode_length]
        else:
            targets = (np.expand_dims(obs_raw[:, :episode_length], -1) if len(obs_raw.shape) == 4 else obs_raw[:, :episode_length]).copy()
        #targets = targets_raw[:, :episode_length] if (output_mode == RKN.OUTPUT_MODE_POSITIONS or len(obs_raw.shape) == 5) else np.expand_dims(obs_raw[:, :episode_length], -1)
        rs = np.random.RandomState(0 if t else 1)
        obs_valid = rs.rand(num_episodes, episode_length, 1) < 0.5
        obs_valid[:, :5] = True
        print(np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))
        obs[np.logical_not(np.squeeze(obs_valid))] = 0
        #np.concatenate([np.ones([num_episodes, 5, 1], dtype=np.bool),
                    #                np.zeros([num_episodes, episode_length - 5, 1], dtype=np.bool)], 1)
        factors = obs_valid.astype(np.float32)
        #obs_valid =
    else:
        raise AssertionError("Invalid Task Mode")
    return obs, targets, obs_valid, factors

train_obs, train_targets, train_obs_valid, train_factors = prepare_data(train_episodes, False)
test_obs, test_targets, test_obs_valid, test_factors = prepare_data(test_episodes, True)

m_print("... data successfully generated")
tf.logging.warn(name + " data successfully generated")

if "rkn" in model_type:
    p = PendulumPlotter(plot_path=name + "Plots", file_name_prefix="initial", plot_n=0)
#    p.plot_transition_matrix(model.transition_matrix.eval(session=model_runner.tf_session))

#vid_gen = VideoGeneration("vid_data", 20)

def bce(targets, predictions):
    targets = targets.astype(np.float64) / 255.0
    epsilon = 1e-8
    point_wise_error = - (targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
    sample_wise_error = np.sum(point_wise_error, axis=(-3, -2, -1))
    return np.mean(sample_wise_error)


for j in range(training_intervals):
    print("Interval", j + 1)
    tf.logging.warn(name + " Interval " + str(j+1))

    model_runner.train(observations=train_obs,
                       targets=train_targets, #projected_train_targets if output_mode == RKN.OUTPUT_MODE_POSITIONS else train_observations,
                       training_epochs=1,
                       observations_valid=train_obs_valid)
    """Evaluation"""
    model_runner.evaluate(observations=test_obs,
                          targets=test_targets,
                          observation_valid=test_obs_valid)

    if plotting:
        print("Generating Plots")
        p = PendulumPlotter(plot_path=name + "Plots", file_name_prefix="Iteration" + str(j + 1), plot_n=5)

        if not model_type in ["lstm", "gru", "llrkn"] and task_mode == RKN.TASK_MODE_FILTER:
            m_print("Initial Covariance:", model.initial_state_covar.eval(session=model_runner.tf_session))
            if not state_dep_trans_covar:
                m_print("Transition Noise Upper:", model.transition_covar_upper.eval(session=model_runner.tf_session))
                m_print("Transition Noise Lower:", model.transition_covar_lower.eval(session=model_runner.tf_session))



            add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                         model.post_mean, model.post_covar,
                         model.prior_mean, model.prior_covar,
                         model.transition_covar, model.kalman_gain]

            debug_length = 8
        else:
            add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                         model.post_mean, model.post_covar]
            debug_length = 4
        if use_likelihood:
            add_fetch.append(model.v_predictions)

        predictions, debug_info = model_runner.predict(test_obs, observation_valid=test_obs_valid, #HERE!!
                                                       additional_fetch=add_fetch)


        #vid_gen.save_vid_to(inputs=test_obs, predictions=predictions, targets=test_targets, folder="iter"+str(j))
        if use_likelihood:
            variance = debug_info[debug_length]
        else:
            variance = None
        if output_mode == RKN.OUTPUT_MODE_POSITIONS:
            if variance is not None:
                p.plot_histogram(test_targets, predictions, variance)
            p.plot_predictions(predictions, test_targets[:10], colors=test_factors[:10], variance=variance[:10])
            save_dict = {"targets": test_targets,
                         "predictions": predictions,
                         "variance": variance,
                         "factors": test_factors}
            np.savez_compressed(name + "Plots/Iteration" + str(j + 1) + ".npz", **save_dict)

        else:
            p.plot_observation_sequences(test_targets, predictions, test_obs[:10] * (test_obs_valid[:10,:, :, np.newaxis, np.newaxis]).astype(np.float32))
            if variance is not None and output_mode == RKN.OUTPUT_MODE_POSITIONS:
                p.plot_histogram(test_targets, predictions, variance)
                p.plot_variance(variance=variance, noise_facts=test_factors)
        p.plot_diagnostics(*debug_info[:debug_length], noise_factors=None if multiple_pendulums else test_factors)

        if model_type[:3] == "rkn":
            tm = model.transition_matrix.eval(session=model_runner.tf_session)
            p.plot_transition_matrix(tm)
#        p.plot_loss_per_time(test_targets[:10], predictions, variance[:10])
        p.close_all()
        predictions[np.squeeze(test_obs_valid)] = (test_targets[np.squeeze(test_obs_valid)].astype(np.float32) / 255)
        print("Imputation loss", bce(test_targets, predictions))
