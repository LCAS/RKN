import os
num_exp = 1
seed_offset = 0
cluster = False

def launch_experiment(model_type,
                      transition_matrix,
                      use_ll,
                      training_intervals,
                      fix_ll_decoder,
                      latent_obs_dim,
                      decoder_mode,
                      state_dep_trans_covar,
                      indi_trans_covar,
                      adapt_covar_to_norm,
                      n_step_pred,
                      immediate_pred,
                      train_noisy,
                      sigmoid_before_norm):

    for i in range(num_exp):
        seed = seed_offset + 2 * i

        command = (("ezex run -nvd -tl '" if cluster else "python3 ") + "rkn/runPendulumConstantNoiseExperiments.py" +
                   " --model_type " + model_type +
                   (" --transition_matrix " + transition_matrix if transition_matrix  != "" else "") +
                   (" --use_ll" if use_ll else "") +
                   (" --fix_l_decoder" if fix_ll_decoder else "") +
                   " --latent_obs_dim " + str(latent_obs_dim) +
                   " --training_intervals " + str(training_intervals) +
                   " --seed " + str(seed) +
                   " --decoder_mode " + decoder_mode +
                   (" --adapt_covar" if adapt_covar_to_norm else "") +
                   (" --state_dep_trans_covar" if state_dep_trans_covar else "") +
                   (" --indi_trans_covar" if indi_trans_covar else "") +
                   " --n_step_pred " + str(n_step_pred) +
                   (" --immediate_pred" if immediate_pred else "")+
                   (" --train_noisy" if train_noisy else "" )+
                   (" --sigmoid_before_norm" if sigmoid_before_norm else "")+
                   ("'" if cluster else ""))
        if cluster:
            command = command + (" Ex_" + model_type + "_" +
                                (transition_matrix + "_" if model_type[0:3] == "rkn" else "") +
                                ((str(latent_obs_dim) + "_") if latent_obs_dim != 50 else "" )+
                                "decMode_" + decoder_mode + "_" +
                                ("adaptCovar_" if adapt_covar_to_norm else "") +
                                ("stateDepTransCovar_" if state_dep_trans_covar else "") +
                                ("indiTransCovar_" if indi_trans_covar else "") +
                                 "nStepPred_" + str(n_step_pred) + "_" +
                                 ("immediatePred_" if immediate_pred else "") +
                                 ("trainNoisy_" if train_noisy else "") +
                                 ("SigmoidBeforeNorm_" if sigmoid_before_norm else "") +
                                str(i+1))


        print(command)
        os.system(command)

"""lstm"""
#for use_ll in [True, False]:
train_intervals = 20
for dec_mode in ["lin", "nonlin"]:
    launch_experiment(model_type="rknc", transition_matrix="band_sd", use_ll=True, training_intervals=train_intervals,
                      fix_ll_decoder=False, latent_obs_dim=50, decoder_mode=dec_mode, state_dep_trans_covar=False,
                      indi_trans_covar=False, adapt_covar_to_norm=False, n_step_pred=0, immediate_pred=False, train_noisy=False,
                      sigmoid_before_norm=False)
