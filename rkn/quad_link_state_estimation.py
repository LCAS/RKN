from data.NLinkPendulum import NLinkPendulum
from rkn.RKN import RKN
import numpy as np
from util.LayerNormalization import LayerNormalization
from tensorflow import keras as k

with_noise = False

# CAVEAT: I DID NOT VERIFY YET THAT THIS REPRODUCES THE RESULTS STATED IN THE PAPER (due to lack of resources)
# WORKING ON IT


class QuadLinkStateEstemRKN(RKN):

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
        return [k.layers.Dense(units=10, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=10, activation=k.activations.relu)]


n_link = NLinkPendulum(episode_length=150,
                       train_episodes=4000,
                       test_episodes=1000,
                       pendulum=NLinkPendulum.QL,
                       generate_img_noise=with_noise,
                       keep_clean_imgs=False,
                       friction=0.1 * np.ones(4),
                       dt=0.05,
                       seed=42)


train_obs, train_targets = n_link.train_images, n_link.to_sc_representation(n_link.train_angles)
test_obs, test_targets = n_link.test_images, n_link.to_sc_representation(n_link.test_angles)

# Build Model
rkn = QuadLinkStateEstemRKN(latent_observation_dim=100, output_dim=8, num_basis=15, bandwidth=3, never_invalid=True)
inputs=k.layers.Input(shape=(None,)+train_obs.shape[2:]) #specify input dimensions, None indicating variable timestep size
rkn(inputs=inputs) 
rkn.compile(optimizer=k.optimizers.Adam(clipnorm=5.0), loss=rkn.gaussian_nll, metrics=[rkn.rmse])

# Train Model
rkn.fit(train_obs, train_targets, batch_size=25, epochs=1000, validation_data=(test_obs, test_targets))
