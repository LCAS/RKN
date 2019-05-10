import os

num_exp = 1
seed_offset = 0
cluster = False


def launch_experiment(pend_type,
                      model_type,
                      latent_obs_dim,
                      decoder_mode,
                      bw,
                      reg_loss_fact):

    for i in range(num_exp):
        command = (("ezex run -nvd -tl '" if cluster else "python3 ") + "rkn/runNLinkPendulumExperiment.py" +
                   " --pend_type " + pend_type +
                   " --model_type " + model_type +
                   " --bandwidth " + str(bw) +
                   " --latent_obs_dim " + str(latent_obs_dim) +
                   " --decoder_mode " + decoder_mode +
                   ("'" if cluster else ""))
        if cluster:
            command = command + (" Ex_" + pend_type + "_"
                                 + model_type + "_" +
                                "decMode_" + decoder_mode + "_" +
                                "lod_" + str(latent_obs_dim) + "_" +
                                "bw_" + str(bw) + "_" +
                                str(i+1))
        print(command)
        os.system(command)

lods = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700,  800 , 900, 1000]
bws = [5, 15, 25]
for bw in bws:
    for lod in lods:
        launch_experiment(pend_type="ql", model_type="rknc", latent_obs_dim=lod, decoder_mode="nonlin",
                          reg_loss_fact=0.0, bw=bw)