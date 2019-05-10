from model import RKN, RKNRunner
from config.PendulumConstantNoiseConfig import PendulumConstantNoiseConfig
from baselines import LSTMBaseline, LSTMBaselineRunner
from model_local_linear.LLRKN import LLRKN
from data import Pendulum
import numpy as np
from plotting.PendulumPlotting import PendulumPlotter
import tensorflow as tf

"""Flags"""

flags = tf.app.flags
flags.DEFINE_string("name", "model", "name of the model")
#flags.DEFINE_string("task_mode", "filter", "task mode of the model, either 'filter' or 'predict'")
#flags.DEFINE_float("pendulum_friction", 0.1, "Friction factor of the pendulum")
flags.DEFINE_bool("use_ll", True, "Whether to use the LL or the MSE to optimize")
flags.DEFINE_bool("fix_l_decoder", False, "Whether to fix the likelihood decoder to the mean or not")
flags.DEFINE_integer("latent_obs_dim", 50, "Latent observation dimension")
flags.DEFINE_string("model_type", "rknc", "which model to use, either 'rkns', 'rknc', 'llrkn' or 'lstm'")
flags.DEFINE_string("transition_matrix", "band_sd_step", "which type of transition matrix to use in case of rkn, either 'band', 'sd' (spring damper), or 'ssd' (stable spring damper)")
flags.DEFINE_bool("state_dep_trans_covar", False, "Whether to learn a state dependent transition covar")
flags.DEFINE_bool("indi_trans_covar", False, "Whether to learn a individual value for each entry of the transition covar")
flags.DEFINE_bool("adapt_covar", False, "Whether to adapt the covar to normalization or not")
#flags.DEFINE_integer("train_episodes", 200, "Number of epochs to use for training")
flags.DEFINE_integer("seed", 0, "random seed for the data generator")
flags.DEFINE_integer("training_intervals", 10, "number of intervals to train for")
#flags.DEFINE_bool("corr_trans_var", False, "Whether to learn a correlated Transition Covariance or not")
flags.DEFINE_string("decoder_mode", "lin", "Which decoder to use, currently 'lin' and 'nonlin' supported")
#flags.DEFINE_bool("single_enc_var", False, "Whether to learn a single encoder variance.")
#flags.DEFINE_integer("give_first_n", 5, "How many obs to give in case of prediction")
flags.DEFINE_integer("n_step_pred", 0, "How many steps to predict into the future during filter (negative = 0)")
flags.DEFINE_float("reg_loss_fact", 0, "")
flags.DEFINE_bool("immediate_pred", False, "Whether to intermediate predict n steps into future or iterating pred step")
flags.DEFINE_bool("train_noisy", False, "Whether to use the noisy targets or not")
flags.DEFINE_bool("sigmoid_before_norm", False, "Whether to use a sigmoid before the normalization")

"""Configure"""

name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)

m_print("Starting Pendulum Experiment")

latent_observation_dim = flags.FLAGS.latent_obs_dim
m_print("Latent observation dimension", latent_observation_dim)

#if flags.FLAGS.task_mode == RKN.TASK_MODE_FILTER:
#    task_mode = RKN.TASK_MODE_FILTER
#    m_print("Task mode: Filter")
#elif flags.FLAGS.task_mode == RKN.TASK_MODE_PREDICT:
#    task_mode = RKN.TASK_MODE_PREDICT
#    m_print("Task mode: Predict")
#else:
#    raise AssertionError("Invalid Task Mode")

model_type = flags.FLAGS.model_type
m_print("Using", model_type, "as transition cell")

transition_matrix = flags.FLAGS.transition_matrix
if transition_matrix == "band":
    transition_matrix = RKN.TRANS_MATRIX_BAND
    m_print("Using band matix")
elif transition_matrix == "sd":
    transition_matrix = RKN.TRANS_MATRIX_SPRING_DAMPER
    m_print("Using spring damper")
elif transition_matrix == "ssd":
    transition_matrix = RKN.TRANS_MATRIX_STABLE_SPRING_DAMPER
    m_print("Using stable spring damper ")
elif transition_matrix == "band_sd_lin":
    transition_matrix = RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH
    m_print("Using Band Spring Damper matrix, linear init")
elif transition_matrix == "band_sd_step":
    transition_matrix = RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_STEP
    m_print("Using Band Spring Damper matrix, step init")
else:
    raise AssertionError("Invalid Transition matrix")

try:
    seed_offset = int(name[-1]) - 1
except:
    seed_offset = 0
data_seed = flags.FLAGS.seed + seed_offset
m_print("Data Seed", data_seed)


pendulum_friction = 0.1 # flags.FLAGS.pendulum_friction
m_print("Pendulum Friction", pendulum_friction)

use_ll = flags.FLAGS.use_ll
m_print("Using Negative LogLikelihood", use_ll)

fix_l_decoder = flags.FLAGS.fix_l_decoder
m_print("Fixing Likelihood decoder", fix_l_decoder)

if flags.FLAGS.decoder_mode == "nonlin":
    decoder_mode = RKN.DECODER_MODE_NONLINEAR
    m_print("Using nonlnear decoder")
elif flags.FLAGS.decoder_mode == "lin":
    decoder_mode = RKN.DECODER_MODE_LINEAR
    m_print("Using linear decoder")
else:
    raise AssertionError("Invalid Decoder Mode")

training_intervals = flags.FLAGS.training_intervals
m_print("Training for ", training_intervals, "intervals")

state_dep_trans_covar = flags.FLAGS.state_dep_trans_covar
m_print("Learning state dependent transition covar:", state_dep_trans_covar)

indi_trans_covar = flags.FLAGS.indi_trans_covar
m_print("Learning individual values ofr state covariance entries:", indi_trans_covar)

n_step_pred = np.maximum(flags.FLAGS.n_step_pred, 0)
m_print("Prediction", n_step_pred, "into future")

train_noisy = flags.FLAGS.train_noisy
m_print("Train with noisy targets:", train_noisy)

immediate_pred = flags.FLAGS.immediate_pred
m_print("Immediate_pred:", immediate_pred)

adapt_covar_to_norm = flags.FLAGS.adapt_covar
if transition_matrix == RKN.TRANS_MATRIX_BAND:
    m_print("Adapt Covar To normalization:", adapt_covar_to_norm)

train_episodes = 200#flags.FLAGS.train_episodes
m_print("Train Episodes:", train_episodes)




use_sigmoid_in_normalization = flags.FLAGS.sigmoid_before_norm
m_print("Use sigmoid with normalization:", use_sigmoid_in_normalization)

#if task_mode == RKN.TASK_MODE_PREDICT:
#    give_first_n = flags.FLAGS.give_first_n
#    m_print("give first ", give_first_n, "observations")


obs_dim = 24
seq_length = 150 #if task_mode == RKN.TASK_MODE_FILTER else 50

""" Build data generator"""

data_config = Pendulum.pendulum_default_params()
data_config[Pendulum.FRICTION_KEY] = pendulum_friction

data_gen = Pendulum(img_size=obs_dim,
                    observation_mode=Pendulum.OBSERVATION_MODE_BALL,
                    transition_noise_std=0.1,
                    observation_noise_std=0.1,
                    pendulum_params=data_config,
                    seed=data_seed)

""" Build Model"""

config = PendulumConstantNoiseConfig(name,
                                     task_mode=RKN.TASK_MODE_FILTER,
                                     model_type=model_type,
                                     transition_matrix=transition_matrix,
                                     use_constant_observation_covariance=False,
                                     use_likelihood=use_ll,
                                     fix_likelihood_decoder=fix_l_decoder,
                                     latent_observation_dim=latent_observation_dim,
                                     decoder_mode=decoder_mode,
                                     state_dependent_transition_covar=state_dep_trans_covar,
                                     individual_transition_covar=indi_trans_covar,
                                     adapt_var_to_norm=adapt_covar_to_norm,
                                     n_step_prediction=n_step_pred if not immediate_pred else 0,
                                     use_sigmoid_in_normalization=False)

if model_type[:3] == "rkn":
    model = RKN(config, debug_recurrent=True)
    model_runner = RKNRunner(model)
elif model_type == "lstm":
    model = LSTMBaseline(config)
    model_runner = LSTMBaselineRunner(model)
elif model_type == "llrkn":
    model = LLRKN(config, debug_recurrent=True)
    model_runner = RKNRunner(model)

"""Generate Data"""

def sample_and_prepare(num_episodes, sequence_length):
    obs, targets,  _, targets_noisy = data_gen.sample_data_set(num_episodes, sequence_length + n_step_pred, full_targets=False)
#    obs, targets,  _, targets_noisy = data_gen.sample_data_set(num_episodes, sequence_length)
    obs_valid = None
    #obs = np.expand_dims(obs, -1)
#    targets = targets[:, n_step_pred: sequence_length + n_step_pred]
#    targets_noisy = targets_noisy[:, n_step_pred: sequence_length + n_step_pred]
   # if task_mode == RKN.TASK_MODE_FILTER:
    obs = np.expand_dims(obs[:, :sequence_length], -1)
    targets = targets[:, n_step_pred: sequence_length + n_step_pred]
    targets_noisy = targets_noisy[:, n_step_pred: sequence_length + n_step_pred]
    #elif task_mode == RKN.TASK_MODE_PREDICT:
    #    obs = obs[:, :give_first_n]
    #    obs_valid = np.concatenate([np.ones([num_episodes, give_first_n, 1], dtype=np.bool),
    #                                np.zeros([num_episodes, sequence_length - give_first_n, 1], dtype=np.bool)], 1)
    return obs, targets, targets_noisy, obs_valid

#train_episodes = train_episodes # 3 * train_episodes if task_mode == RKN.TASK_MODE_PREDICT else train_episodes
train_obs, train_targets, train_targets_noisy, train_obs_valid = sample_and_prepare(train_episodes, seq_length)
test_obs, test_targets, test_targets_noisy, test_obs_valid = sample_and_prepare(200, seq_length)

m_print("Baseline:", np.sqrt(np.mean(np.square(test_targets-test_targets_noisy))))

"""Train and evaluate"""

train_epochs = 50

for i in range(training_intervals):
    print("Iteration", (i + 1))
    if train_noisy:
        model_runner.train(observations=train_obs, targets=train_targets_noisy, training_epochs=train_epochs)
        model_runner.evaluate(test_obs, test_targets_noisy)
        model_runner.evaluate(test_obs, test_targets)
    else:
        model_runner.train(observations=train_obs, targets=train_targets, training_epochs=train_epochs,
                           observations_valid=train_obs_valid)
        model_runner.evaluate(test_obs, test_targets, observation_valid=test_obs_valid)

    p = PendulumPlotter(plot_path=name + "Plots", file_name_prefix="Iteration" + str(i + 1), plot_n=5)
    if not model_type == "lstm":
     #  if not  fix_initial_state_covar:
        m_print("Initial Covariance:", model.initial_state_covar.eval(session=model_runner.tf_session))
    #    if not fix_trans_covar:
        if not state_dep_trans_covar:
            m_print("Transition Noise Upper:", model.transition_covar_upper.eval(session=model_runner.tf_session))
            m_print("Transition Noise Lower:", model.transition_covar_lower.eval(session=model_runner.tf_session))
        print("Generating Plots")

        add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                     model.post_mean, model.post_covar,
                     model.prior_mean, model.prior_covar,
                     model.transition_covar, model.kalman_gain]
        debug_length = 8
    else:
        add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
         model.post_mean, model.post_covar]
        debug_length = 4
    if use_ll:
            add_fetch.append(model.v_predictions)

    predictions, debug_info = model_runner.predict(test_obs, additional_fetch=add_fetch)
    if use_ll:
        variance = debug_info[debug_length]
    else:
        variance = None
      #  if train_noisy:
      #      p.plot_predictions(predictions, test_targets_noisy, variance)
      #  else:
    p.plot_predictions(predictions, test_targets, variance)
    if train_noisy:
        p.plot_smoothing(test_targets_noisy, test_targets, predictions)
    p.plot_diagnostics(*debug_info[:debug_length])
    if "rkn" in model_type:
        tm = model.transition_matrix.eval(session=model_runner.tf_session)
        p.plot_transition_matrix(tm)

    p.plot_loss_per_time(test_targets, predictions, variance)
    p.close_all()
