from data import Pendulum
from network import HiddenLayersParamsKeys, NDenseHiddenLayers, SimpleOutputLayer, FeedForwardNet, NConvolutionalHiddenLayers, HiddenLayerWrapper
import tensorflow as tf
import numpy as np
import util.AngularRepresentation as ar
from util.LossFunctions import mse

dim = 2
obs_dim = 24
num_seqs = 200
seq_length = 150

data_len = num_seqs * seq_length
data_gen = Pendulum(img_size=obs_dim,
                    observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1,
                    observation_noise_std=0.1)
train_obs, _,  _, train_ts_obs = data_gen.sample_data_set(num_seqs, seq_length)

conv_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 2,
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

dense_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 2,
              HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
              HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
              HiddenLayersParamsKeys.WIDTH_PREFIX + '1': 50,
              HiddenLayersParamsKeys.WIDTH_PREFIX + '2': 10}

inputs = tf.placeholder(shape=[None, obs_dim, obs_dim, 1], dtype=tf.float32 )
targets = tf.placeholder(shape=[None, dim], dtype=tf.float32)
encoder_conv_hidden = NConvolutionalHiddenLayers(conv_dict, "enc_h_conv", flatten_output=True, on_gpu=False, up_convolutional=False)
encoder_dense_hiddden = NDenseHiddenLayers(dense_dict, "enc_h_dense")
encoder_out = SimpleOutputLayer(dim, "enc_out")

encoder = FeedForwardNet(output_layer=encoder_out, hidden_layers=HiddenLayerWrapper([encoder_conv_hidden, encoder_dense_hiddden]))
predictions = encoder(inputs)

loss = mse(targets, predictions)
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_obs = np.reshape(train_obs, [data_len, obs_dim, obs_dim, 1])
train_ts_obs = np.reshape(ar.to_angular_representation(train_ts_obs), [data_len, dim])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
batch_size = 500
batches_per_epoch = int(np.floor(data_len / batch_size))
for i in range(epochs):
    avg_loss = 0
    rand_idx = np.random.permutation(data_len)

    for j in range(batches_per_epoch):
        batch_slice = slice(j * batch_size, (j+1) * batch_size)
        current_obs = train_obs[rand_idx[batch_slice]]
        current_targets = train_ts_obs[rand_idx[batch_slice]]

        _, cur_loss = sess.run(fetches=(optimizer, loss), feed_dict={inputs:  current_obs, targets: current_targets})
        avg_loss += cur_loss / batches_per_epoch
    print("Epoch", i, "Loss", avg_loss)


test_obs, _,  _, test_ts_obs = data_gen.sample_data_set(num_seqs, seq_length)
test_ts_obs = np.reshape(ar.to_angular_representation(test_ts_obs), [data_len, dim])
test_obs = np.reshape(test_obs, [data_len, obs_dim, obs_dim, 1])
print("Evaluation Loss", sess.run(loss, feed_dict={inputs: test_obs, targets:test_ts_obs}))







