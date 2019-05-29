import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from data.PendulumData import Pendulum
from rkn.RKN import RKN
from util.LayerNormalization import LayerNormalization


def generate_imputation_data_set(pendulum, num_seqs, seq_length, seed):
    obs, _, _, _ = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs = np.expand_dims(obs, -1)
    targets = obs.copy()

    rs = np.random.RandomState(seed=seed)
    obs_valid = rs.rand(num_seqs, seq_length, 1) < 0.5
    obs_valid[:, :5] = True
    print("Fraction of Valid Images:", np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))
    obs[np.logical_not(np.squeeze(obs_valid))] = 0

    return obs, obs_valid, targets


# Implement Encoder and Decoder hidden layers
class PendulumImageImputationRKN(RKN):

    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(12, kernel_size=5, padding="same"),
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
            k.layers.Dense(30, activation=k.activations.relu)]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(144, activation=k.activations.relu),
            k.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 3, 16])),

            k.layers.Conv2DTranspose(16, kernel_size=5, strides=4, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),

            k.layers.Conv2DTranspose(12, kernel_size=3, strides=2, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu)
        ]


# Generate Data
pend_params = Pendulum.pendulum_default_params()
pend_params[Pendulum.FRICTION_KEY] = 0.1
data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                transition_noise_std=0.1,
                observation_noise_std=1e-5,
                seed=0,
                pendulum_params=pend_params)

train_obs, train_obs_valid, train_targets = generate_imputation_data_set(data, 1000, 150, seed=42)
test_obs, test_obs_valid, test_targets = generate_imputation_data_set(data, 250, 150, seed=23541)

# Build Model
rkn = PendulumImageImputationRKN(latent_observation_dim=15, output_dim=train_targets.shape[-3:],
                                 num_basis=15, bandwidth=3, never_invalid=False)
input1=k.layers.Input(shape=(None,)+train_obs.shape[2:])#specify input dimensions, None indicating variable timestep size
input2=k.layers.Input(shape=(None,1)) #specify input dimensions, None indicating variable timestep size
inputs=(input1,input2) #multi inputs passed as tuple
rkn(inputs)

rkn.compile(optimizer=k.optimizers.Adam(clipnorm=5.0),
            loss=lambda t, p: rkn.bernoulli_nll(t, p, uint8_targets=True))

# Train Model
rkn.fit((train_obs, train_obs_valid),
        train_targets, batch_size=25, epochs=1000,
        validation_data=((test_obs, test_obs_valid), test_targets))








