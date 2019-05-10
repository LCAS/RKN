import os

num_exp = 1
seed_offset = 0
cluster = False

def launch_experiment(model_type,
                      latent_obs_dim,
                      decoder_mode):

    for i in range(num_exp):
        command = (("ezex run -nvd -tl '" if cluster else "python3 ") + "rkn/runToyBlockExperiments.py" +
                   " --model_type " + model_type +
                   " --latent_obs_dim " + str(latent_obs_dim) +
                   " --decoder_mode " + decoder_mode +
                   ("'" if cluster else ""))
        if cluster:
            command = command + (" Ex_" + model_type + "_" +
                                "decMode_" + decoder_mode + "_" +
                                "latentObsDim_" + str(latent_obs_dim) + "_" +
                                str(i+1))
        print(command)
        os.system(command)


for dec_mode in ["lin", "nonlin"]:
    launch_experiment(model_type="rkn", latent_obs_dim=150, decoder_mode=dec_mode)
    for latent_obs_dim in [150, 25]:
        for model in ["lstm", "gru"]:
            launch_experiment(model_type=model, latent_obs_dim=latent_obs_dim, decoder_mode=dec_mode)