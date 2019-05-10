import os
num_exp = 1
seed_offset = 0
cluster = False

def launch_experiment(model_type,
                      output_mode,
                      task_mode,
                      use_likelihood,
                      band_width,
                      num_basis,
                      latent_obs_dim,
                      training_intervals,
                      norm_latent,
                      decoder_mode):

    for i in range(num_exp):
        seed = seed_offset + 2 * i

        command = (("ezex run -nvd -tl'" if cluster else "python3 ") + "rkn/runPendulumImageNoiseExperiments.py" +
                   " --output_mode " + output_mode +
                   " --task_mode " + task_mode +
                   " --band_width " + str(band_width) +
                   " --num_basis " + str(num_basis) +
                   " --model_type " + model_type +
                   (" --norm_latent" if norm_latent else "") +
                   (" --use_ll" if use_likelihood else "")+
                   " --latent_obs_dim " + str(latent_obs_dim) +
                   " --seed " + str(seed) +
                   " --training_intervals " + str(training_intervals) +
                   " --decoder_mode " + decoder_mode +
                  ("'" if cluster else ""))
        if cluster:
            command = command + (" Ex_" + task_mode[0] + output_mode[0] + "_" + model_type + "_" +
                                ((str(latent_obs_dim) + "_") if latent_obs_dim != 50 else "" ) +
                                "bw" + str(band_width) + "_nb" + str(num_basis) + "_"
                                "decMode_" + decoder_mode + "_" + ("normed_" if norm_latent else "") +
                                 ("ll_" if use_likelihood else "") +
                                 str(i+1))


        print(command)
        os.system(command)

for bw in [0, 1, 3, 5, 10, 15]:
    launch_experiment(output_mode="obs", task_mode="pred", model_type="llrkn", latent_obs_dim=15,
                      training_intervals=20, decoder_mode="nonlin", norm_latent=False,
                      use_likelihood=False, band_width=bw, num_basis=15)
#     launch_experiment(output_mode="pos", task_mode="filt", model_type="gru", latent_obs_dim=14,
#                          training_intervals=30, decoder_mode=decmode, norm_latent=False,
#                          use_likelihood=use_likelihood, band_width=3, num_basis=15)
#        launch_experiment(output_mode="pos", task_mode="filt", model_type="llrkn", latent_obs_dim=15,
#                          training_intervals=20, decoder_mode=decmode, norm_latent=False,
#                          use_likelihood=use_likelihood, band_width=15, num_basis=15)

    #launch_experiment(output_mode="pos", task_mode="pred", model_type="lstm", latent_ob
    # s_dim=6,
    #                  training_intervals=20, decoder_mode="nonlin", norm_latent=False,
    #                  use_likelihood="True", band_width=3, num_basis=1)
    #launch_experiment(output_mode="pos", task_mode="pred", model_type="llrkn", latent_obs_dim=15,
    #                  training_intervals=20, decoder_mode="nonlin", norm_latent=False,
    #                  use_likelihood=ll, band_width=3, num_basis=15)


"""
#for init in ["fix", "rand"]:
    #launch_experiment(model_type="rknc", transition_matrix="band_smooth", latent_obs_dim=50, training_intervals=20,
    #                  decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=True, norm_post=True,
    #                  norm_prior=True, reg_loss_fact=0.0, trans_init=init, indi_trans_covar=True)
    #launch_experiment(model_type="rknc", transition_matrix="band_smooth", latent_obs_dim=50, training_intervals=20,
    #                  decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=False, norm_post=False,
    #                  norm_prior=False, reg_loss_fact=0.0, trans_init=init, indi_trans_covar=True)
    #launch_experiment(model_type="rknc", transition_matrix="band_smooth", latent_obs_dim=50, training_intervals=20,
    #                  decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=True, norm_post=True,
    #                  norm_prior=True, reg_loss_fact=1.0, trans_init=init, indi_trans_covar=True)
 #   launch_experiment(model_type="rknc", transition_matrix="band_smooth", latent_obs_dim=50, training_intervals=20,
  #                    decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=True, norm_post=True,
   #                   norm_prior=True, reg_loss_fact=0.0, trans_init=init, indi_trans_covar=False)
#launch_experiment(model_type="rknc", transition_matrix="band_sd", latent_obs_dim=50, training_intervals=20,
#                  decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=False, norm_post=False,
#                  norm_prior=False, reg_loss_fact=0.0, trans_init="fix", indi_trans_covar=False)
#launch_experiment(model_type="rknc", transition_matrix="band_sd", latent_obs_dim=50, training_intervals=20,
#                  decoder_mode="lin", tc_init_upper=0.1, tc_init_lower=0.1, norm_latent=True, norm_post=True,
#                  norm_prior=True, reg_loss_fact=0.0, trans_init="fix", indi_trans_covar=False)
"""