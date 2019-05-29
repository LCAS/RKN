from rkn.RKN import RKN
from data.NLinkPendulum import NLinkPendulum
import numpy as np
from util.LayerNormalization import LayerNormalization
from tensorflow import keras as k
import tensorflow as tf

# CAVEAT: I DID NOT VERIFY YET THAT THIS REPRODUCES THE RESULTS STATED IN THE PAPER (due to lack of resources)
# WORKING ON IT


def generate_imputation_data_set(quad_link, seed, test_data=False):
    obs = quad_link.test_images if test_data else quad_link.train_images
    targets = obs.copy()

    rs = np.random.RandomState(seed=seed)
    obs_valid = rs.rand(obs.shape[0], obs.shape[1], 1) < 0.5
    obs_valid[:, :5] = True
    print("Fraction of Valid Images:", np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))
    obs[np.logical_not(np.squeeze(obs_valid))] = 0

    return obs, obs_valid, targets


class QuadLinkImageImputationRKN(RKN):

    def build_encoder_hidden(self):
        # 1: Conv Layer
        return [
            k.layers.Conv2D(12, kernel_size=5, padding="same", strides=2),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            k.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(200, activation=k.activations.relu)
        ]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(144, activation=k.activations.relu),
            k.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 3, 16])),

            k.layers.Conv2DTranspose(16, kernel_size=5, strides=4, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),

            k.layers.Conv2DTranspose(12, kernel_size=3, strides=4, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu)
        ]


n_link = NLinkPendulum(episode_length=150,
                       train_episodes=100,
                       test_episodes=100,
                       pendulum=NLinkPendulum.QL,
                       generate_img_noise=False,
                       keep_clean_imgs=False,
                       friction=0.1 * np.ones(4),
                       dt=0.05,
                       seed=42)

train_obs, train_obs_valid, train_targets = generate_imputation_data_set(n_link, seed=42, test_data=False)
test_obs, test_obs_valid, test_targets = generate_imputation_data_set(n_link, seed=421, test_data=True)

# Build Model
rkn = QuadLinkImageImputationRKN(observation_shape=train_obs.shape[-3:], latent_observation_dim=100,
                                 output_dim=train_targets.shape[-3:], num_basis=15, bandwidth=3)
rkn.compile(optimizer=k.optimizers.Adam(clipnorm=5.0),
            loss=lambda t, p: rkn.bernoulli_nll(t, p, uint8_targets=True))

# Train Model
rkn.fit((train_obs, train_obs_valid),
        train_targets, batch_size=25, epochs=1000,
        validation_data=((test_obs, test_obs_valid), test_targets))
