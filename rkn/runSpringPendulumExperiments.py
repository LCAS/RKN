"""Not designed for use on cluster - also not necessary"""
from data.SpringPendulum import SpringPendulum
from model.RKNLinear import RKNLinear
from transition_cell.TransitionCell import TransitionCell
import numpy as np
import matplotlib.pyplot as plt
from config.SpringPendulumConfig import SpringPendulumConfig

"""Parameters"""
num_seqs = 10
seq_length = 100
state_covar_model = "corr"

if state_covar_model == "full":
    transition_cell_type = TransitionCell.TRANSITION_CELL_FULL
elif state_covar_model == "simple":
    transition_cell_type = TransitionCell.TRANSITION_CELL_SIMPLE
elif state_covar_model == "corr":
    transition_cell_type = TransitionCell.TRANSITION_CELL_CORRELATED
else:
    raise AssertionError("Invalid state covar model")

use_likelihood = True
learn_transition_model = False
learn_transition_covar = False
learn_observation_covar = False
learn_initial_covar = False
training_epochs = 0

"""Data Generator"""
dim = 2
true_transition_std = 0.01
true_observation_std = 0.1
data_gen = SpringPendulum(dim=dim,
                          transition_covar=true_transition_std**2,
                          observation_covar=true_observation_std**2)

transition_std_init = [0.01, 0.01, 0.0]
observation_std_init = 0.1

if transition_cell_type == TransitionCell.TRANSITION_CELL_FULL:
    transition_covar = data_gen.transition_covar
    observation_covar = data_gen.observation_covar
    initial_state_covar = np.eye(dim * 2)
elif transition_cell_type == TransitionCell.TRANSITION_CELL_SIMPLE:
    transition_covar = np.diag(data_gen.transition_covar)
    observation_covar = np.diag(data_gen.observation_covar)
    initial_state_covar = np.ones(dim * 2)
elif transition_cell_type == TransitionCell.TRANSITION_CELL_CORRELATED:
    transition_covar = np.concatenate([np.diag(data_gen.transition_covar),
                                       np.diag(data_gen.transition_covar[dim:, :dim])], 0)
    observation_covar = np.diag(data_gen.observation_covar)
    initial_state_covar = np.concatenate([np.ones(dim * 2), np.zeros(dim)], 0)
else:
    raise AssertionError("invalid transition cell type")


"""Config"""
config = SpringPendulumConfig(name="model",
                              dim = dim,
                              use_likelihood=use_likelihood,
                              transition_cell_type=transition_cell_type,
                              transition_matrix=None if learn_transition_model else data_gen.transition_matrix,
                              transition_covariance=None if learn_transition_covar else transition_covar,
                              transition_covariance_init= [x**2 for x in transition_std_init] if learn_transition_covar else None,
                              initial_state_covariance=None if learn_initial_covar else initial_state_covar,
                              initial_state_covariance_init=1.0 if learn_initial_covar else None)


"""Model"""
model = RKNLinear(config,
                  observation_covar=None if learn_observation_covar else observation_covar,
                  observation_covar_init=observation_std_init**2 if learn_observation_covar else None,
                  debug_recurrent=False)

"""Train"""
train_states, train_obs = data_gen.sample_sequences(num_seqs, seq_length)
model.train(train_obs, train_states[:, :, 0:dim], training_epochs)


print("True Transition Matrix")
print(data_gen.transition_matrix)

print("Learned Transition Matrix")
print(model.eval_tensor(model.transition_matrix))

test_states, test_obs = data_gen.sample_sequences(num_seqs, seq_length)

model.evaluate(test_obs, test_states[:, :, 0:dim])

filtered_means_rkn, filtered_covars_rkn = model.filter(test_obs)
filtered_means_gtkf, filtered_covars_gtkf = data_gen.get_kf_gt(test_obs)

print("RKN ALL", np.sqrt(np.mean(np.square(filtered_means_rkn - test_states))))
print("RKN POS", np.sqrt(np.mean(np.square(filtered_means_rkn[:, :, 0:dim] - test_states[:, :, 0:dim]))))

def ll(prediction, target, variance):
    element_wise_ll = 0.5 * (((prediction - target)**2) / variance + np.log(variance) + np.log(2 * np.pi))
    sample_wise_ll = np.sum(element_wise_ll, -1)
    return np.mean(sample_wise_ll)

def full_nll(prediction, target, variance):
    dim = prediction.shape[-1]
    constant_term = dim * np.log(2 * np.pi)
    reg_term = np.log(np.linalg.det(variance))
    diff = prediction - target
    loss_term = np.tensordot(diff, np.linalg.solve(variance, diff), axes=[[2], [2]])
    return np.mean(0.5 * (constant_term + reg_term + loss_term))


print("KF ALL", np.sqrt(np.mean(np.square(filtered_means_gtkf - test_states))))
print("KF POS", np.sqrt(np.mean(np.square(filtered_means_gtkf[:, :, 0:dim] - test_states[:, :, 0:dim]))))
print("KF POS LL", ll(filtered_means_gtkf[:, :, 0:dim], test_states[:, :, 0:dim], np.diagonal(filtered_covars_gtkf, axis1=-2, axis2=-1)[:, :, :dim]))
print("KF POS Full LL", full_nll(filtered_means_gtkf[:, :, 0:dim], test_states[:, :, 0:dim], filtered_covars_gtkf[:, :, :dim, :dim]))

print("RKN MinVar", np.min(np.diagonal(filtered_covars_rkn, axis1=-2, axis2=-1)))
print("KF MinVar", np.min(np.diagonal(filtered_covars_gtkf, axis1=-2, axis2=-1)))



fig = plt.figure()
full_covariances = transition_cell_type == TransitionCell.TRANSITION_CELL_FULL

if dim == 1:
    ax = plt.subplot(4, 1, 1)
    plt.plot(test_obs[0, :, 0], c='black')
    plt.plot(test_states[0, :, 0], c='blue')
    plt.plot(filtered_means_gtkf[0, : ,0], c='green')
    plt.plot(filtered_means_rkn[0, :, 0], c='red')
    plt.legend(["input", "true", "kaman filter", "rkn"])

    ax = plt.subplot(4, 1, 2)
    plt.plot(test_states[0, :, 1], c='blue')
    plt.plot(filtered_means_gtkf[0, : ,1], c='green')
    plt.plot(filtered_means_rkn[0, :, 1], c='red')

    ax = plt.subplot(4, 1, 3)
    plt.plot(filtered_covars_gtkf[0, : ,0, 0], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 0, 0], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 0], c='red')

    ax = plt.subplot(4, 1, 4)
    plt.plot(filtered_covars_gtkf[0, : ,1, 1], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 1, 1], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 1], c='red')

elif dim == 2:
    ax = plt.subplot(8, 1, 1)
    plt.plot(test_obs[0, :, 0], c='black')
    plt.plot(test_states[0, :, 0], c='blue')
    plt.plot(filtered_means_gtkf[0, :, 0], c='green')
    plt.plot(filtered_means_rkn[0, :, 0], c='red')
    plt.legend(["input", "true", "kaman filter", "rkn"])

    ax = plt.subplot(8, 1, 2)
    plt.plot(test_obs[0, :, 1], c='black')
    plt.plot(test_states[0, :, 1], c='blue')
    plt.plot(filtered_means_gtkf[0, :, 1], c='green')
    plt.plot(filtered_means_rkn[0, :, 1], c='red')

    ax = plt.subplot(8, 1, 3)
    plt.plot(test_states[0, :, 2], c='blue')
    plt.plot(filtered_means_gtkf[0, :, 2], c='green')
    plt.plot(filtered_means_rkn[0, :, 2], c='red')

    ax = plt.subplot(8, 1, 4)
    plt.plot(test_states[0, :, 3], c='blue')
    plt.plot(filtered_means_gtkf[0, :, 3], c='green')
    plt.plot(filtered_means_rkn[0, :, 3], c='red')

    ax = plt.subplot(8, 1, 5)
    plt.plot(filtered_covars_gtkf[0, :, 0, 0], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 0, 0], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 0], c='red')

    ax = plt.subplot(8, 1, 6)
    plt.plot(filtered_covars_gtkf[0, :, 1, 1], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 1, 1], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 1], c='red')

    ax = plt.subplot(8, 1, 7)
    plt.plot(filtered_covars_gtkf[0, :, 2, 2], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 2, 2], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 2], c='red')

    ax = plt.subplot(8, 1, 8)
    plt.plot(filtered_covars_gtkf[0, :, 3, 3], c='green')
    if full_covariances:
        plt.plot(filtered_covars_rkn[0, :, 3, 3], c='red')
    else:
        plt.plot(filtered_covars_rkn[0, :, 3], c='red')

plt.show()






