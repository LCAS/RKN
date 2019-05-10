from config.ToyBlockConfig import ToyBlockTowerConfig
from model.RKN import RKN
from model.RKNRunner import RKNRunner
from data.ToyBlockTowerData import ToyBlockTowerData
from plotting.TBTPlotting import TBTPlotter
from baselines import GRUBaseline, LSTMBaseline, LSTMBaselineRunner, GRUBaselineRunner
import util.GPUUtil as gpu_util
import os
import tensorflow as tf
import numpy as np
#Todo Plotting

""" For developing - overwrites flags!"""
# Todo remove before "release"

"""Configuration"""
flags = tf.app.flags

flags.DEFINE_string("name", "model", "name of the model")
#flags.DEFINE_string('output_mode', 'positions', "Either 'positions' or 'observations'")
flags.DEFINE_string("model_type", "rkn", "transition model used, either the 'rkn' (default) or one of the baselines 'lstm' or 'gru'")
flags.DEFINE_string("decoder_mode", "lin", "which decoder model to use")
flags.DEFINE_integer("latent_obs_dim", 150, "dimensionality of latent observations")

""" Configuration"""
name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)

m_print("Starting Pendulum Experiment")

model_type = flags.FLAGS.model_type

output_mode = RKN.OUTPUT_MODE_POSITIONS
#if flags.FLAGS.output_mode == 'observations':
#    print("Output Mode: observations")
#    output_mode = RKN.OUTPUT_MODE_OBSERVATIONS
#elif flags.FLAGS.output_mode == 'positions':
#    print("Output Mode: positions")
#    output_mode = RKN.OUTPUT_MODE_POSITIONS
#else:
#    raise AssertionError("invalid output mode flag - needs to be either 'positions' or 'observations'")

iterations = 8
epochs_per_iteration = 25
max_seqs = 6000 # size of train set

if flags.FLAGS.decoder_mode == 'lin':
    m_print("Linear Decoder")
    decoder_mode = RKN.DECODER_MODE_LINEAR
elif flags.FLAGS.decoder_mode == 'nonlin':
    m_print("Nonlinear Decoder")
    decoder_mode = RKN.DECODER_MODE_NONLINEAR
else:
    raise AssertionError("Invalid Decoder mode")

latent_observation_dim = flags.FLAGS.latent_obs_dim
m_print("Latent Observation Dim", latent_observation_dim)
seq_length = 15
gpu_available = gpu_util.get_num_gpus() > 0
# if gpu available we are on the cluster.
base_path = "/work/scratch/ix98zuvy/ToyBlockTowerData" if gpu_available else "/home/philipp/RKN/toyBlockData"

"""model"""
config = ToyBlockTowerConfig(name=name,
                             output_mode=output_mode,
                             latent_observation_dim=latent_observation_dim,
                             decoder_mode=decoder_mode)
if model_type == 'rkn':
    model = RKN(config=config, debug_recurrent=True)
    model_runner = RKNRunner(model)
elif model_type == 'lstm':
    model = LSTMBaseline(config=config)
    model_runner = LSTMBaselineRunner(model)
elif model_type == 'gru':
    model = GRUBaseline(config=config)
    model_runner = GRUBaselineRunner(model=model)
else:
    raise AssertionError("Invalid tranistion Model")

"""data"""
data = ToyBlockTowerData(train_set_path=os.path.join(base_path, "train"),
                         test_set_path=os.path.join(base_path, "test"),
                         img_size=config.input_dim[:2],
                         max_seqs=max_seqs,
                         seq_length=seq_length,
                         with_gt=output_mode==RKN.OUTPUT_MODE_POSITIONS)

def prepare_data(data_fn):
    if output_mode == RKN.OUTPUT_MODE_POSITIONS:
        observations, full_states = data_fn()
        targets = np.concatenate([full_states[:, :, 0:3], full_states[:, :,12:15], full_states[:, :, 6:9]], -1)
    else:
        observations = data_fn()
        targets = observations

    num_seqs = len(observations)
    obs_valid = np.ones([num_seqs, seq_length, 1], dtype=np.bool)
                               # np.zeros([num_seqs, seq_length - give_first_n, 1], dtype=np.bool)], 1)

    return observations, targets, obs_valid


"""train"""
train_observations, train_targets, train_obs_valid = prepare_data(data_fn=data.get_train_data)
test_observations, test_targets, test_obs_valid = prepare_data(data_fn=data.get_test_data)

for i in range(iterations):
    model_runner.train(observations=train_observations,
                       targets=train_targets,
                       training_epochs=epochs_per_iteration,
                       observations_valid=train_obs_valid)

    """eval"""
    model_runner.evaluate(observations=test_observations,
                          targets=test_targets,
                          observation_valid=test_obs_valid,
                          test_batch_size=config.batch_size)

    print("Generating Plots")
    p = TBTPlotter(plot_path=name + "Plots", file_name_prefix="Iteration" + str(i + 1), plot_n=5)

    if not model_type in ["lstm", "gru"]:
        m_print("Initial Covariance:", model.initial_state_covar.eval(session=model_runner.tf_session))
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
    add_fetch.append(model.v_predictions)

    predictions, debug_info = model_runner.predict(test_observations, observation_valid=test_obs_valid,
                                                   additional_fetch=add_fetch)
    variance = debug_info[debug_length]
    p.plot_predictions(predictions, test_targets)

    p.plot_variance(variance=variance)
    p.plot_diagnostics(*debug_info[:debug_length])

    #if model_type[:3] == "rkn":
    #    tm = model.transition_matrix.eval(session=model_runner.tf_session)
    #    p.plot_transition_matrix(tm)
    p.plot_loss_per_time(test_targets, predictions, variance)
    p.close_all()


