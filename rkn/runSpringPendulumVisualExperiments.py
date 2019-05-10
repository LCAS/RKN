from data.SpringPendulum import SpringPendulum
from config.SpringPendulumVisualConfig import SpringPendulumVisualConfig
from baselines import LSTMBaseline, LSTMBaselineRunner
from model_local_linear.LLRKN import LLRKN
import numpy as np
from model import RKN, RKNRunner
from transition_cell import TransitionCell

dim = 1
true_transition_std = 0.01
true_observation_std = 0.1
obs_dim = 200

model_type = "lstm"



learn_transition_matrix = False
learn_transition_covar = False
learn_observation_covar = False
learn_initial_covar = False

transition_cell_type = TransitionCell.TRANSITION_CELL_CORRELATED
transition_std_init = [0.01, 0.01, 0.0]
observation_std_init = 0.1

initial_covariance_init = 1.0

data_gen = SpringPendulum(dim=dim,
                          transition_covar=true_transition_std**2,
                          observation_covar=true_observation_std**2)

train_states, train_ts_obs = data_gen.sample_sequences(100, 100)
train_obs = data_gen.generate_images(train_ts_obs, dim=obs_dim)

if transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
    transition_covar = np.diag(data_gen.transition_covar)
    observation_covar = np.diag(data_gen.observation_covar)
elif transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
    transition_covar = np.concatenate([np.diag(data_gen.transition_covar),
                                       np.diag(data_gen.transition_covar[dim:, :dim])], 0)
    observation_covar = np.diag(data_gen.observation_covar)
else:
    raise AssertionError("invalid transition cell type")
initial_state_covar = 1.0

config = SpringPendulumVisualConfig(name="model",
                                    observation_dim=obs_dim,
                                    transition_cell_type=transition_cell_type,
                                    transition_matrix=None if learn_transition_matrix else data_gen.transition_matrix,
                                    transition_covar=None if learn_transition_covar else transition_covar,
                                    transition_covar_init=[x**2 for x in transition_std_init] if learn_transition_covar else None,
                                    observation_covar=None if learn_observation_covar else observation_covar,
                                    observation_covar_init=observation_std_init**2 if learn_observation_covar else None,
                                    initial_covar=None if learn_initial_covar else initial_state_covar,
                                    initial_covar_init=initial_covariance_init if learn_initial_covar else None,
                                    model_type=model_type)

if model_type == "rkn":
    model = RKN(config, debug_recurrent=True)
    model_runner = RKNRunner(model)
elif model_type == "lstm":
    model = LSTMBaseline(config)
    model_runner = LSTMBaselineRunner(model)

model_runner.train(train_obs, train_states[:, :, 0:1], 1000)

test_states, test_ts_obs = data_gen.sample_sequences(100, 100)
test_obs = data_gen.generate_images(test_ts_obs, dim=obs_dim)

model_runner.evaluate(test_obs, test_states[:, :, 0:1])