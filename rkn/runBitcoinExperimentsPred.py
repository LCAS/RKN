from data.BitcoinDataHandling import DataManager
from model import RKN, RKNRunner
from baselines import LSTMBaseline, LSTMBaselineRunner, GRUBaseline, GRUBaselineRunner
import numpy as np
import matplotlib.pyplot as plt
from config.BitcoinConfig import BitcoinConfig

DATA_PATH = "/home/philipp/Code/dataset_features_btc_relative.pkl"
TRANSITION_MODEL = "lstm" # either RKN, LSTM or GRU
LOSS_FN = "bce" # either likelihood, rmse or bce (last one only for boolean targets)
USE_BOOLEAN_TARGETS = True
batch_size = 500

#Todo check of those values are correct..
data = DataManager(DATA_PATH, norm_method="siraj_norm", target_window_size=24,
                   prediction_offset=12, input_length=192, boolean_targets=USE_BOOLEAN_TARGETS)

train_features, train_targets = data.train_data_batched
test_features, test_targets = data.test_data_batched

train_features[:, 168:, :] = 0
test_features[:, 168:, :] = 0


obs_valid_train = np.concatenate([np.ones([train_targets.shape[0], 168, 1], dtype=np.bool),
                                  np.zeros([train_targets.shape[0], 24, 1], dtype=np.bool)] , -2)

obs_valid_test = np.concatenate([np.ones([test_targets.shape[0], 168, 1], dtype=np.bool),
                                 np.zeros([test_targets.shape[0], 24, 1], dtype=np.bool)], -2)


#train_features = np.nan_to_num(train_features)

#for i in range(10):
#    nr = np.random.randint(0, test_targets.shape[0])
#    plt.figure()
#    plt.plot(test_targets[nr, :, 0], c="blue")
#    plt.plot(train_targets[nr, :, 0], c="green")

#plt.show()

config = BitcoinConfig(num_features=6,
                       latent_observation_dim=25,
                       batch_size=batch_size,
                       bptt_length=train_features.shape[1],
                       loss_fn=LOSS_FN)

if TRANSITION_MODEL == "rkn":
    model = RKN(name="model",
                config=config)
    model_runner = RKNRunner(model)

elif TRANSITION_MODEL == "lstm":
    model = LSTMBaseline(name="model",
                         config=config)
    model_runner = LSTMBaselineRunner(model)
elif TRANSITION_MODEL == "gru":
    model = GRUBaseline(name="model",
                        config=config)
    model_runner = GRUBaselineRunner(model)
else:
    raise AssertionError("Invalid transition model")


if not USE_BOOLEAN_TARGETS:
    print("Train Baseline:", np.sqrt(np.mean((train_features[:, :, 0:1] - train_targets)**2)))
    print("Test Baseline:", np.sqrt(np.mean((test_features[:, :, 0:1] - test_targets)**2)))

for i in range(3):
    model_runner.train(observations=train_features[:, :168, :],
                       targets=train_targets,
                       observations_valid=obs_valid_train,
                       training_epochs=1)
    model_runner.evaluate(observations=test_features[:, :168, :],
                          targets=test_targets,
                          observation_valid=obs_valid_test)
    test_pred = model_runner.predict(test_features[:, :168, :], observation_valid=obs_valid_test)
    if USE_BOOLEAN_TARGETS:
        pred = model_runner.predict(observations=test_features)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        print("Test Acc", np.count_nonzero(pred == test_targets) / np.prod(test_targets.shape))
        print("Test Acc Final", np.count_nonzero(pred[:, 168:, : ] == test_targets[:, 168:, :]) / (pred.shape[0] * 24))
    else:
        print("Test Loss", np.sqrt(np.mean((test_pred[:, 168:, :] - test_targets[:, 168:, :])**2)))



test_pred = model_runner.predict(test_features)
for i in range(10):

    nr = np.random.randint(0, test_pred.shape[0])
    fig = plt.figure()
    if USE_BOOLEAN_TARGETS:
        axis =  fig.gca()
        axis.set_ylim(-0.25, 1.25)
        plt.plot(np.arange(0, 192), .5*np.ones(192), c="black")
    plt.plot(test_targets[nr, :, 0], c="blue")
    plt.plot(np.arange(0, 168), test_pred[nr, :168, 0], c="green")
    plt.plot(np.arange(168, 192), test_pred[nr, 168:, 0], c="red")

plt.show()
