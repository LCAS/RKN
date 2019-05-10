from data.KittiData import KittiData
from config.KittiConfig import KittiConfig
from util import GPUUtil
import tensorflow as tf
import numpy as np
from model import RKNSimpleTransitionCell, RKNRunner, RKN

flags = tf.app.flags
flags.DEFINE_string("name", "model", "the name of the model")
flags.DEFINE_string("kitti_mode", "temp", "mode, either 'temp' or 'stereo'")
flags.DEFINE_integer("latent_obs_dim", 10, "Dimensionality of latent obervations")
flags.DEFINE_integer("bandwidth", 2, "Bandwidth of transition model")

name = flags.FLAGS.name
def m_print(*args):
    print(name + ":", *args)
m_print("Starting Kitti Experiment")

if flags.FLAGS.kitti_mode == "temp":
    kitti_mode = KittiData.KITTI_MODE_TEMPORAL_PAIR
elif flags.FLAGS.kitti_mode == "stereo":
    kitti_mode = KittiData.KITTI_MODE_STEREO_CURRENT
else:
    raise AssertionError("Invalid Kitti mode - needs to be either 'temp' or 'stereo'")
m_print("Data Mode", kitti_mode)

seq_length = 50
m_print("Sequence Length", seq_length)
test_set = 10
m_print("Testing on sequence", test_set)
latent_observation_dim = flags.FLAGS.latent_obs_dim
m_print("Latent Observation Dim", latent_observation_dim)
bandwidth = flags.FLAGS.bandwidth
m_print("Bandwidth", bandwidth)
init_mode = RKNSimpleTransitionCell.INIT_MODE_RANDOM
transition_noise_std_model = 0.02

num_iterations = 10
steps_per_iteration = 25

# If gpu available we are on the cluster
base_path = "/work/scratch/ix98zuvy/KittiData" if GPUUtil.get_num_gpus() > 0 else "/home/philipp/Code/KittiData"

data = KittiData(base_path=base_path, seq_length=50, mode=KittiData.KITTI_MODE_STEREO_CURRENT, test_seqs=[10])
batch_size = int(np.floor(data.num_train_batches / 4))
m_print("Batch Size", batch_size)
config = KittiConfig(latent_observation_dim=latent_observation_dim,
                     bandwidth=bandwidth,
                     init_mode=init_mode,
                     batch_size=batch_size,
                     seq_length=seq_length,
                     transition_noise_covar=transition_noise_std_model**2)
model = RKN(name=name, config=config)
model_runner = RKNRunner(model)

for i in range(num_iterations):
    model_runner.train(observations=data.train_obs,
                       targets=data.train_pos,
                       training_epochs=steps_per_iteration)

    model_runner.evaluate(observations=data.test_obs,
                          targets=data.test_pos)

