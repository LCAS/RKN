from e2c.E2C import E2C
from data.PendulumData import Pendulum
import numpy as np
from network.HiddenLayers import HiddenLayersParamsKeys
from util.GPUUtil import get_num_gpus
from util.LossFunctions import binary_crossentropy
from plotting import PendulumPlotter
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("name", "model", "name of the model")
name = flags.FLAGS.name


seed = 0
img_dim = 24
obs_dim = [2 * img_dim, img_dim, 1]

train_episodes = 1000
test_episodes = 500
sequence_length = 150
latent_dim = 3

pend_params = Pendulum.pendulum_default_params()
pend_params[Pendulum.FRICTION_KEY] = 0.1

data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                transition_noise_std=0.1,
                observation_noise_std=1e-5,
                seed=seed,
                pendulum_params=pend_params)

def prepare_data(observations):
    num_batches, seq_length = observations.shape[:2]
    seq_length = seq_length - 2
    obs_t0 = np.zeros([num_batches, seq_length] + obs_dim, dtype=np.uint8)
    obs_t1 = np.zeros([num_batches, seq_length] + obs_dim, dtype=np.uint8)
    for i in range(num_batches):
        for j in range(seq_length):
            obs_t0[i, j, :img_dim] = observations[i, j]
            obs_t0[i, j, img_dim:] = observations[i, j + 1]
            obs_t1[i, j, :img_dim] = observations[i, j * 1]
            obs_t1[i, j, img_dim:] = observations[i, j + 2]
    return obs_t0, obs_t1

train_obs = np.expand_dims(data.sample_data_set(train_episodes, sequence_length + 2, False, seed=42)[0], -1)
train_obs_t0, train_obs_t1 = prepare_data(train_obs)

test_obs = np.expand_dims(data.sample_data_set(test_episodes, sequence_length + 2, False, seed=1337)[0], -1)
test_obs_t0, test_obs_t1 = prepare_data(test_obs)


encoder_conv_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                     HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                     HiddenLayersParamsKeys.PADDING: 'SAME',
                     HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                     HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                     HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
                     HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 12,
                     HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 1,
                     HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '1': 2,
                     HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '1': 2,

                     HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 3,
                     HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 12,
                     HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 2,
                     HiddenLayersParamsKeys.POOL_SIZE_PREFIX + '2': 2,
                     HiddenLayersParamsKeys.POOL_STRIDE_PREFIX + '2': 2}

encoder_dense_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                      HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                      HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                      HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 50}

decoder_dense_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                      HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                      HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                      HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 288}

decoder_conv_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                     HiddenLayersParamsKeys.LAYER_NORMALIZATION: True,
                     HiddenLayersParamsKeys.PADDING: 'SAME',
                     HiddenLayersParamsKeys.POOL_FN: tf.layers.max_pooling2d,
                     HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,

                     HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '1': 5,
                     HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '1': 16,
                     HiddenLayersParamsKeys.STRIDES_PREFIX + '1': 4,

                     HiddenLayersParamsKeys.FILTER_SIZE_PREFIX + '2': 3,
                     HiddenLayersParamsKeys.OUT_CHANNELS_PREFIX + '2': 12,
                     HiddenLayersParamsKeys.STRIDES_PREFIX + '2': 2}

decoder_initial_shape = [6, 3, 16]

transition_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 2,
                   HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                   HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                   HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 100,
                   HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 100}

model = E2C(observation_dim=obs_dim,
            action_dim=0,
            latent_dim=latent_dim,
            learning_rate=1e-3,
            factor_regularization_loss=1.0,
            factor_distance_loss=0.25,
            encoder_conv_dict=encoder_conv_dict,
            encoder_dense_dict=encoder_dense_dict,
            decoder_conv_dict=decoder_conv_dict,
            decoder_dense_dict=decoder_dense_dict,
            decoder_init_shape=decoder_initial_shape,
            use_gpu=False,
            transition_dict=transition_dict,
            reconstruction_loss_function=lambda t, p: binary_crossentropy(t, p, img_data=True, scale_targets=255))

for i in range(8):
    model.train(np.reshape(train_obs_t0, [-1] + obs_dim), np.reshape(train_obs_t1, [-1] + obs_dim), batch_size=3750, training_epochs=25)
    model.evaluate(np.reshape(test_obs_t0, [-1] + obs_dim), np.reshape(test_obs_t1, [-1] + obs_dim), test_batch_size=3750)

    obs_valid = np.random.rand(test_episodes, sequence_length) < 0.5
    obs_valid[:, :5] = True
    print(np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))

    p = PendulumPlotter(plot_path=name + "Plots", file_name_prefix="Iteration" + str(i + 1), plot_n=5)
    avg_imp_loss = 0
    avg_rec_loss = 0
    seq_l = []

    for i in range(test_episodes):
        imp_loss, rec_loss, seq = model.impute_sequence(test_obs_t0[i], obs_valid[i], test_obs_t1[i], img_size=img_dim)
        seq_l.append(seq)
        avg_imp_loss += imp_loss
        avg_rec_loss += rec_loss
    avg_imp_loss = avg_imp_loss / test_episodes
    avg_rec_loss = avg_rec_loss / test_episodes
    print("Imputation Loss", avg_imp_loss, "Reconstruction loss", avg_rec_loss)

    p.plot_observation_sequences(inputs=(test_obs_t0[:10].astype(np.float32) / 255) * np.reshape(obs_valid[:10], (10, sequence_length, 1, 1, 1)),
                             predictions=np.concatenate([np.expand_dims(s, 0) for s in seq_l], 0)[:10],
                             targets=test_obs_t0[:10].astype(np.float32) / 255)
