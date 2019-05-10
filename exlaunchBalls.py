import os

num_exp = 1
seed_offset = 0
cluster = False

def launch_experiment(dynamics,
                      model_type,
                      transition_matrix,
                      give_tm,
                      latent_obs_dim,
                      norm_latent,
                      norm_obs_only,
                      decoder_mode,
                      reg_loss_fact):

    for i in range(num_exp):
        seed = seed_offset + 2 * i

        command = (("ezex run -nvd -tl '" if cluster else "python3 ") + "rkn/runBallExperiments.py" +
                   " --dynamics " + dynamics +
                   " --model_type " + model_type +
                   (" --transition_matrix " + transition_matrix if transition_matrix  != "" else "") +
                   (" --give_tm" if give_tm else "") +
                   (" --norm_latent" if norm_latent else "") +
                   (" --norm_obs_only" if norm_obs_only else "") +
                   " --seed " + str(seed) +
                   " --latent_obs_dim " + str(latent_obs_dim) +
                   " --decoder_mode " + decoder_mode +
                   " --reg_loss_fact " + str(reg_loss_fact) +
                   ("'" if cluster else ""))
        if cluster:
            command = command + (" Ex_" + dynamics[0] + "_"
                                  + model_type + "_" +
                                (transition_matrix + "_" if model_type[0:3] == "rkn" else "") +
                                "decMode_" + decoder_mode + "_" +
                                "latent_obs_dim_" + str(latent_obs_dim) + "_" +
                                str(i+1))
        print(command)
        os.system(command)


for dynamics in ["quad", "double"]:
    launch_experiment(dynamics=dynamics, model_type='rknc', transition_matrix='band_smooth', give_tm=False, latent_obs_dim=100,
                      decoder_mode="nonlin", norm_latent=True, norm_obs_only=True, reg_loss_fact=0.0)
    launch_experiment(dynamics=dynamics, model_type="lstm", transition_matrix='band', give_tm=False,
                      decoder_mode="nonlin", norm_latent=True, norm_obs_only=True, reg_loss_fact=0.0, latent_obs_dim=100)
    launch_experiment(dynamics=dynamics, model_type="gru", transition_matrix='band', give_tm=False,
                      decoder_mode="nonlin", norm_latent=False, norm_obs_only=False, reg_loss_fact=0.0, latent_obs_dim=100)
    launch_experiment(dynamics=dynamics, model_type="lstm", transition_matrix='band', give_tm=False,
                      decoder_mode="nonlin", norm_latent=True, norm_obs_only=True, reg_loss_fact=0.0, latent_obs_dim=8)
    launch_experiment(dynamics=dynamics, model_type="gru", transition_matrix='band', give_tm=False,
                      decoder_mode="nonlin", norm_latent=False, norm_obs_only=False, reg_loss_fact=0.0, latent_obs_dim=10)






