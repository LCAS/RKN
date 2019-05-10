from data.NLinkPendulum import NLinkPendulum
from config.NLinkPendulumConfig import NLinkPendulumConfig
import tensorflow as tf
from model.RKN import RKN
from model_local_linear.LLRKN import LLRKN
from model.RKNRunner import RKNRunner
from baselines import LSTMBaselineRunner, LSTMBaseline, GRUBaselineRunner, GRUBaseline
from plotting.NLinkPendPlotting import NLinkPendulumPlotter
import numpy as np
from util.GPUUtil import get_num_gpus
from plotting.VideoGeneration import VideoGeneration

flags = tf.app.flags

flags.DEFINE_string("name", "model", "name of the model")
flags.DEFINE_string("output_mode", "pos", "either pos or obs")
flags.DEFINE_string("task_mode", "filt", "either pred or filt")
flags.DEFINE_string("pend_type", "ql", "either 'dl' or 'ql'")
flags.DEFINE_string("model_type", "rknc", "which model to use, either 'rkns', 'rknc', 'gru' or 'lstm'")
flags.DEFINE_string("decoder_mode", "nonlin", "Which decoder to use, currently 'lin' and 'nonlin' supported")
flags.DEFINE_integer("latent_obs_dim", 1000, "dimensionality of latent observations")
flags.DEFINE_integer("bandwidth", 25, "bandwidth of transition matrices")
#flags.DEFINE_float("reg_loss_fact", 0.0, "")
#flags.DEFINE_bool("add_img_noise", False, "")

name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)

model_type = flags.FLAGS.model_type
m_print("Using", model_type, "as transition cell")

if flags.FLAGS.decoder_mode == "nonlin":
    decoder_mode = RKN.DECODER_MODE_NONLINEAR
    m_print("Using nonlnear decoder")
elif flags.FLAGS.decoder_mode == "lin":
    decoder_mode = RKN.DECODER_MODE_LINEAR
    m_print("Using linear decoder")
else:
    raise AssertionError("Invalid Decoder Mode")

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

band_width = flags.FLAGS.bandwidth
m_print("Bandwidth", band_width)

latent_observation_dim = flags.FLAGS.latent_obs_dim
m_print("Latent Observation Dim", latent_observation_dim)

#reg_loss_factor = flags.FLAGS.reg_loss_fact
#m_print("Regularization loss factor", reg_loss_factor)

if flags.FLAGS.pend_type == "dl":
    pendulum_type = NLinkPendulum.DL
elif flags.FLAGS.pend_type == "ql":
    pendulum_type = NLinkPendulum.QL
else:
    raise AssertionError("Invalid Pendulum Type")

add_img_noise = False #(pendulum_type == NLinkPendulum.DL)
m_print("Add image noise:", add_img_noise)


train_episodes = 4000
episode_length = 150
test_episodes = 1000
intervals = 12
epochs_per_interval = 25

config = NLinkPendulumConfig(name=name,
                             task_mode=task_mode,
                             output_mode=output_mode,
                             num_links= 2 if pendulum_type == NLinkPendulum.DL else 4,
                             model_type=model_type,
                             bw=band_width,
                             latent_observation_dim=latent_observation_dim,
                             decoder_mode=decoder_mode,
                             reg_loss_factor=0.0)

if model_type[:3] == 'rkn':
    model = RKN(config=config, debug_recurrent=False)
    model_runner = RKNRunner(model)
    #model_runner.initialize_logger("logs", track_runtime=True)
elif model_type == 'llrkn':
    model = LLRKN(config=config, debug_recurrent=False)
    model_runner = RKNRunner(model)
elif model_type == 'lstm':
    model = LSTMBaseline(config=config)
    model_runner = LSTMBaselineRunner(model)
elif model_type == 'gru':
    model = GRUBaseline(config=config)
    model_runner = GRUBaselineRunner(model=model)
else:
    raise AssertionError("Invalid tranistion Model")


data = NLinkPendulum(episode_length=episode_length,
                     train_episodes=train_episodes,
                     test_episodes=test_episodes,
                     pendulum=pendulum_type,
                     generate_img_noise=add_img_noise,
                     keep_clean_imgs=False,
                     friction=0.1 * (np.ones(2) if model == NLinkPendulum.DL else np.ones(4)),
                     dt=0.05)

#base_path = "/home/philipp/RKN/kvae/" if get_num_gpus() == 0 else "/work/scratch/ix98zuvy/"
#train_data = np.load(base_path + "data/ql.npz")
#train_obs = train_data['images']
#data =

train_obs = data.train_images
test_obs = data.test_images
#test_data = np.load(base_path + "data/ql_test.npz")

#test_obs = test_data['images']
test_factors = data.test_factors

if output_mode == RKN.OUTPUT_MODE_POSITIONS:
    train_targets = data.to_sc_representation(data.train_angles)
    test_targets = data.to_sc_representation(data.test_angles)
else:
    train_targets = train_obs.copy()
    test_targets = test_obs.copy()

if task_mode == RKN.TASK_MODE_PREDICT:
    obs_valid_train = np.random.rand(train_episodes, episode_length, 1) < 0.5
    obs_valid_train[:, :5] = True
    print(np.count_nonzero(obs_valid_train) / np.prod(obs_valid_train.shape))
    rs = np.random.RandomState(0)
    obs_valid_test = rs.rand(test_episodes, episode_length, 1) < 0.5
    obs_valid_test[:, :5] = True
    print(np.count_nonzero(obs_valid_test) / np.prod(obs_valid_test.shape))
    train_obs[np.squeeze(np.logical_not(obs_valid_train))] = 0
    test_obs[np.squeeze(np.logical_not(obs_valid_test))] = 0

else:
    obs_valid_train = None
    obs_valid_test = None


#vid_gen = VideoGeneration("vid_data_" + name + "_" + str(pendulum_type)+"_"+model_type)


for i in range(intervals):

    model_runner.train(observations=train_obs, targets=train_targets, training_epochs=epochs_per_interval,
                       observations_valid=obs_valid_train)
    model_runner.evaluate(observations=test_obs, targets=test_targets, test_batch_size=int(config.batch_size/2),
                          observation_valid=obs_valid_test)

    print("Generating Plots")
    p = NLinkPendulumPlotter(plot_path=name + "Plots", file_name_prefix="Iteration" + str(i + 1), plot_n=5)

    #if not model_type in ["lstm", "gru", "llrkn"]:
       # m_print("Initial Covariance:", model.initial_state_covar.eval(session=model_runner.tf_session))
       # m_print("Transition Noise Upper:", model.transition_covar_upper.eval(session=model_runner.tf_session))
       # m_print("Transition Noise Lower:", model.transition_covar_lower.eval(session=model_runner.tf_session))

    #    add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
    #                 model.post_mean, model.post_covar,
    #                 model.prior_mean, model.prior_covar,
    #                 model.transition_covar, model.kalman_gain]
    #    debug_length = 8
    add_fetch = [model.latent_observation_mean, model.latent_observations_covar,
                 model.post_mean, model.post_covar]
    debug_length = 4


#    add_fetch.append(model.v_predictions)

    tov = obs_valid_test[:10] if obs_valid_test is not None else None
    t = test_obs[:10]
    #t[np.squeeze(np.logical_not(tov))] = np.nan
    predictions, debug_info = model_runner.predict(t, observation_valid=tov, additional_fetch=add_fetch)
    #vid_gen.save_vid_to(inputs=test_obs, predictions=predictions, targets=test_targets, folder="iter"+str(i))

#    variance = debug_info[debug_length]
    p.plot_diagnostics(*debug_info[:debug_length], noise_factors=None)

    if output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
        p.plot_observation_sequences(inputs=test_obs[:10], # * np.reshape(obs_valid_test[:10], [10, episode_length, 1, 1, 1]),
                                     targets=test_targets[:10],
                                     predictions=predictions)

    #if output_mode == RKN.OUTPUT_MODE_POSITIONS:
    #    p.plot_predictions(predictions, test_targets[:10], variance=variance[:10])
    #p.plot_variance(variance=variance, noise_facts=test_factors)

    if model_type[:3] == "rkn":
        tm = model.transition_matrix.eval(session=model_runner.tf_session)
        p.plot_transition_matrix(tm)
    p.plot_loss_per_time(test_targets[:10], predictions) #, variance)
    p.close_all()




