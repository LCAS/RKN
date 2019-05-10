from data.SpringPendulum import SpringPendulum
from network import HiddenLayersParamsKeys, NDenseHiddenLayers, SimpleOutputLayer, FeedForwardNet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.LossFunctions import rmse

dim = 1
obs_dim = 200
true_transition_std = 0.01
true_observation_std = 0.1
num_seqs = 100
seq_length = 100
data_len = num_seqs * seq_length
data_gen = SpringPendulum(dim=dim,
                          transition_covar=true_transition_std**2,
                          observation_covar=true_observation_std**2)

_, train_ts_obs = data_gen.sample_sequences(num_seqs, seq_length)
train_obs = data_gen.generate_images(train_ts_obs, dim=obs_dim)

encoder_dict = {HiddenLayersParamsKeys.NUM_LAYERS: 1,
                HiddenLayersParamsKeys.ACTIVATION_FN: tf.nn.relu,
                HiddenLayersParamsKeys.LAYER_NORMALIZATION: False,
                HiddenLayersParamsKeys.KEEP_PROB_PREFIX + "1": 0.8,
                HiddenLayersParamsKeys.WIDTH_PREFIX + "1": 20}

inputs = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32 )
targets = tf.placeholder(shape=[None, 1], dtype=tf.float32)
is_training = tf.placeholder(shape=[], dtype=tf.bool)
encoder_hiddden = NDenseHiddenLayers(encoder_dict, "enc_h")
encoder_out = SimpleOutputLayer(1, "enc_out")

encoder = FeedForwardNet(output_layer=encoder_out, hidden_layers=encoder_hiddden)
predictions = encoder(inputs, is_training=is_training)

loss = rmse(targets, predictions)
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_obs = np.reshape(train_obs, [data_len, obs_dim])
train_ts_obs = np.reshape(train_ts_obs, [data_len, dim])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 100
batch_size = 500
batches_per_epoch = int(np.floor(data_len / batch_size))
for i in range(epochs):
    avg_loss = 0
    rand_idx = np.random.permutation(data_len)

    for j in range(batches_per_epoch):
        batch_slice = slice(j * batch_size, (j+1) * batch_size)
        current_obs = train_obs[rand_idx[batch_slice]]
        current_targets = train_ts_obs[rand_idx[batch_slice]]

        _, cur_loss = sess.run(fetches=(optimizer, loss), feed_dict={inputs:  current_obs, targets: current_targets, is_training: True})

        avg_loss += cur_loss / batches_per_epoch
    print("Epoch", i, "Loss", avg_loss)


_, test_ts_obs = data_gen.sample_sequences(num_seqs, seq_length)
test_obs = data_gen.generate_images(test_ts_obs, dim=obs_dim)
test_ts_obs = np.reshape(test_ts_obs, [data_len, dim])
test_obs = np.reshape(test_obs, [data_len, obs_dim])
print("Evaluation Loss", sess.run(loss, feed_dict={inputs: test_obs, targets:test_ts_obs, is_training: False}))







