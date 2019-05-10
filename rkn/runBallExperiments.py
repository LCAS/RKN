from config.BallTrackingConfig import BallTrackingConfig
from model.RKN import RKN
from baselines import LSTMBaselineRunner, LSTMBaseline, GRUBaselineRunner, GRUBaseline
from model.RKNRunner import  RKNRunner
from transition_cell.RKNSimpleTransitionCell import RKNSimpleTransitionCell
from data.LinearBalls import LinearBalls
from data.QuadLinkBalls import QuadLinkBalls
from data.DoubleLinkBalls import DoubleLinkBalls

import numpy as np
from plotting.BallPloting import BallPlotter
import tensorflow as tf
""" For developing - overwrites flags!"""
# Todo remove before "release"
DEBUG = False
DEBUG_DYNAMICS_MODE = BallTrackingConfig.DYNAMICS_MODE_LINEAR
#DEBUG_DYNAMICS_MODE = BallTrackingConfig.DYNAMICS_MODE_DOUBLE_LINK
#DEBUG_DYNAMICS_MODE = BallTrackingConfig.DYNAMICS_MODE_QUAD_LINK

"""Configure run"""
flags = tf.app.flags

flags.DEFINE_string("name", "model", "name of the model")
flags.DEFINE_string("dynamics", "linear", "Dynamics model to control the balls, either 'linear', 'double' or 'quad'")
flags.DEFINE_string("model_type", "rknc", "transition model used, either the 'rkn' (default) or one of the baselines 'lstm' or 'gru'")
flags.DEFINE_string("transition_matrix", "band", "which type of transition matrix to use in case of rkn, either 'band', 'sd' (spring damper), or 'ssd' (stable spring damper)")
flags.DEFINE_bool("give_tm", False, "Whether to give the true model in case of linear dynamics")
flags.DEFINE_bool("norm_latent", False, "")
flags.DEFINE_bool("norm_obs_only", False, "")
flags.DEFINE_float("reg_loss_fact", 0.0, "")
flags.DEFINE_string("decoder_mode", "nonlin", "Which decoder to use, currently 'lin' and 'nonlin' supported")
flags.DEFINE_string("plot_path", "BallPlots", "path under which the figures are saved")
flags.DEFINE_integer("latent_obs_dim", 100, "")
#
flags.DEFINE_integer("num_plots", 3, "amount of sequences to plot")
flags.DEFINE_integer("seed", 0, "random seed for the data generator")

plotting = False

""" Configuration"""
name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)

m_print("Starting Ball Experiment")
if DEBUG:
    m_print("!!!DEBUGING!!! - flags overwritten")

try:
    seed_offset = int(name[-1]) - 1
except:
    seed_offset = 0
data_seed = flags.FLAGS.seed + seed_offset
m_print("Data Seed", data_seed)

"""Configure Dynamics"""
if DEBUG:
    dynamics_mode = DEBUG_DYNAMICS_MODE
elif flags.FLAGS.dynamics == "linear":
    dynamics_mode = BallTrackingConfig.DYNAMICS_MODE_LINEAR
    m_print("linear dynamics")
elif flags.FLAGS.dynamics == "double":
    dynamics_mode = BallTrackingConfig.DYNAMICS_MODE_DOUBLE_LINK
    m_print("double link dynamics")
elif flags.FLAGS.dynamics == "quad":
    dynamics_mode = BallTrackingConfig.DYNAMICS_MODE_QUAD_LINK
    m_print("quad link dynamics")
else:
    raise AssertionError("Invalid dynamics model, needs to be either 'linear', 'double' or 'quad'")

if flags.FLAGS.decoder_mode == "nonlin":
    decoder_mode = RKN.DECODER_MODE_NONLINEAR
    m_print("Using nonlnear decoder")
elif flags.FLAGS.decoder_mode == "lin":
    decoder_mode = RKN.DECODER_MODE_LINEAR
    m_print("Using linear decoder")
else:
    raise AssertionError("Invalid Decoder Mode")

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
elif transition_matrix == "band_sd":
    transition_matrix = RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_STEP
    m_print("Using Band Spring Damper matrix")
elif transition_matrix == "band_un":
    transition_matrix = RKN.TRANS_MATRIX_BAND_UNNORMAL
    m_print("Using unnormal band transition matrix")
elif transition_matrix == "band_smooth":
    transition_matrix = RKN.TRANS_MATRIX_BAND_SPRING_DAMPER_SMOOTH
    m_print("Using smooth band transition matrix")
else:
    raise AssertionError("Invalid Transition matrix")

"""Configure Transition Model"""
model_type = flags.FLAGS.model_type
m_print("Transition Model:", model_type)

episode_length = 100
m_print("Episode length:", episode_length)

if DEBUG:
    train_episodes = 10
elif dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR:
    train_episodes = 1000
else:
    train_episodes = 4000


train_intervals = 8
m_print("train intervals:", train_intervals)

m_print("train episodes:", train_episodes)
test_episodes = 333 if BallTrackingConfig.DYNAMICS_MODE_LINEAR else 1500
m_print("test episodes:", test_episodes)# per n in [1, max_balls]
max_balls = 10 if DEBUG else (100 if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else 50)
m_print("max balls:", max_balls)
transition_std = 0.002 if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else 0.0
m_print("Transition noise data (std):", transition_std)
training_epochs = 10 if DEBUG else (100 if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else 25)
m_print("Training Epochs:", training_epochs)
normalize_latent = flags.FLAGS.norm_latent
m_print("Normalize Latent:", normalize_latent)
normalize_obs_only = flags.FLAGS.norm_obs_only
m_print("Normalize Obs only:", normalize_obs_only)
give_tm = flags.FLAGS.give_tm
m_print("Giving true transition model:", give_tm)
reg_loss_fact = flags.FLAGS.reg_loss_fact
m_print("Reg Loss Factor:", reg_loss_fact)
latent_obs_dim = flags.FLAGS.latent_obs_dim
m_print("Latent obs dim", latent_obs_dim)

img_size = [64, 64, 3]

plot_path = flags.FLAGS.plot_path
num_plots = 5 #flags.FLAGS.num_plots


"""Data"""
if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR:
    BallData = lambda n_balls, episode_length, train_episodes, test_episodes:\
        LinearBalls(n_balls=n_balls,
                    img_size=img_size,
                    episode_length=episode_length,
                    train_episodes=train_episodes,
                    test_episodes=test_episodes,
                    dyn_sigma=transition_std,
                    seed=data_seed)
elif dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_DOUBLE_LINK:
    BallData = lambda n_balls, episode_length, train_episodes, test_episodes: \
        DoubleLinkBalls(n_balls=n_balls,
                        img_size=img_size,
                        episode_length=episode_length,
                        train_episodes=train_episodes,
                        test_episodes=test_episodes,
                        scale_factor=None,
                        seed=data_seed)
elif dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_QUAD_LINK:
    BallData = lambda n_balls, episode_length, train_episodes, test_episodes:\
        QuadLinkBalls(n_balls=n_balls,
                      img_size=img_size,
                      episode_length=episode_length,
                      train_episodes=train_episodes,
                      test_episodes=test_episodes,
                      scale_factor=None,
                      seed=data_seed)
else:
    raise AssertionError('Invalid Dynamics mode')

data = BallData(n_balls=-max_balls,
                episode_length=episode_length,
                train_episodes=train_episodes,
                test_episodes=test_episodes)

""" Build Model"""
config = BallTrackingConfig(name=name,
                            dynamics_mode=dynamics_mode,
                            latent_obs_dim=latent_obs_dim,
                            model_type=model_type,
                            transition_matrix=transition_matrix,
                            give_linear_transition_matrix=give_tm,
                            decoder_mode=decoder_mode,
                            normalize_latent=normalize_latent,
                            normalize_obs_only=normalize_obs_only,
                            tm=data.transition_matrix if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR else None,
                            reg_loss_factor=reg_loss_fact)
if model_type == 'rknc':
    model = RKN(config=config, debug_recurrent=True)
    model_runner = RKNRunner(model)
elif model_type == 'lstm':
    model = LSTMBaseline(config=config)
    model_runner = LSTMBaselineRunner(model)
elif model_type == 'gru':
    model = GRUBaseline(config=config)
    model_runner = GRUBaselineRunner(model)
else:
    raise AssertionError("Invalid tranistion Model")

train_obs = data.train_observations
train_targets = data.train_positions

test_obs = data.test_observations
test_targets = data.test_positions
test_visibility = data.test_visibility
"""Train"""


"""Prepare Plotter"""
if dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_LINEAR:
    file_name_prefix_suffix = "Linear"
elif dynamics_mode == BallTrackingConfig.DYNAMICS_MODE_DOUBLE_LINK:
    file_name_prefix_suffix = "Double"
else:
    file_name_prefix_suffix = "Quad"

file_name_prefix = name + "Balls" + file_name_prefix_suffix

for i in range(train_intervals):

    model_runner.train(observations=train_obs,
                       targets=train_targets,
                       training_epochs=training_epochs)

    model_runner.evaluate(observations=test_obs, targets=test_targets, test_batch_size=100)

    if plotting:
        plotter = BallPlotter(plot_path=plot_path,
                              file_name_prefix=file_name_prefix + str(i),
                              plot_n=num_plots)


        if not model_type in ["lstm", "gru"]:
            m_print("Initial Covariance:", model.initial_state_covar.eval(session=model_runner.tf_session))
            if not (config.learn_state_dependent_transition_covar or config.transition_covariance_given):
                m_print("Transition Noise Upper:", model.transition_covar_upper.eval(session=model_runner.tf_session))
                m_print("Transition Noise Lower:", model.transition_covar_lower.eval(session=model_runner.tf_session))

            to_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                        model.post_mean, model.post_covar,
                        model.prior_mean, model.prior_covar,
                        model.transition_covar, model.kalman_gain]

            debug_length = 8
        else:
            to_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                        model.post_mean, model.post_covar]
            debug_length = 4

        if config.use_likelihood:
            to_fetch.append(model.v_predictions)

        plotter.plot_transition_matrix(model.transition_matrix.eval(session=model_runner.tf_session))
        predictions, add_fetch = model_runner.predict(observations=test_obs[:num_plots],
                                                      additional_fetch=to_fetch)

     #   plotter.plot_loss_per_time(test_targets, predictions, add_fetch[debug_length] if config.use_likelihood else None)
        plotter.plot_predictions(n_balls=0,
                                 predictions=predictions,
                                 targets=test_targets[:num_plots],
                                 variance=add_fetch[debug_length] if config.use_likelihood else None,
                                 visibility=test_visibility)

        plotter.plot_diagnostics(i, *add_fetch[:debug_length], noise_factors=test_visibility)
        plotter.close_all()

    #m_print("Average Loss: ", np.mean(loss_per_n, 0)[0])


