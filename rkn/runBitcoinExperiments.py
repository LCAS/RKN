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

#Todo check of those values are correct..
data = DataManager(DATA_PATH, data_smoothing=False, norm_method="min_zero_max_one", target_window_size=24,
                   prediction_offset=36, input_length=168, boolean_targets=USE_BOOLEAN_TARGETS)
train_features, train_targets = data.train_data_batched
test_features, test_targets = data.test_data_batched

#train_features = np.nan_to_num(train_features)

#for i in range(10):
#    nr = np.random.randint(0, test_targets.shape[0])
#    plt.figure()
#    plt.plot(test_targets[nr, :, 0], c="blue")
#    plt.plot(train_targets[nr, :, 0], c="green")

#plt.show()

config = BitcoinConfig(num_features=6,
                       latent_observation_dim=25,
                       batch_size=500,
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
else:
    print("Train baseline", np.count_nonzero(train_targets) / np.prod(train_targets.shape))
    print("Test baseline", np.count_nonzero(test_targets) / np.prod(test_targets.shape))

for i in range(10):
    model_runner.train(observations=train_features,
                       targets=train_targets,
                       training_epochs=1)
    model_runner.evaluate(observations=test_features,
                          targets=test_targets)
    if USE_BOOLEAN_TARGETS:
        pred = model_runner.predict(observations=test_features)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        print(np.count_nonzero(pred == test_targets) / np.prod(test_targets.shape))

test_pred = model_runner.predict(test_features)
for i in range(10):
    nr = np.random.randint(0, test_pred.shape[0])
    fig = plt.figure()
    if USE_BOOLEAN_TARGETS:
        axis =  fig.gca()
        axis.set_ylim(-0.25, 1.25)
        plt.plot(.5*np.ones(168), c="black")
    plt.plot(test_targets[nr, :, 0], c="blue")
    plt.plot(test_pred[nr, :, 0], c="green")

plt.show()
