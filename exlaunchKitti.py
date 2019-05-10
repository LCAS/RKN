import os

"""Launches jobs on the cluster (ATTENTION: The number of jobs grows quite fast)
However, this is probably useless for other people anyway since it relies on my modified version of ezex... 
... write your own scripts ;)"""

kitti_modes = ["temp", "stereo"]
bandwidths = [1]#[1, 3]
latent_obs_dims = [5] #[5, 10, 20, 25, 35]
trials = 1 #2

for mode in kitti_modes:
    for bandwidth in bandwidths:
        for latent_obs_dim in latent_obs_dims:
            for trial in range(trials):
                command = ("ezex run -nvd -tl 'rkn/runKittiExperiments" +
                           " --kitti_mode " + mode +
                           " --bandwidth " + str(bandwidth) +
                           " --latent_obs_dim " + str(latent_obs_dim) +
                           "' kitti_" + mode + "_bw" + str(bandwidth) + "_dim" + str(latent_obs_dim))
                print(command)
                os.system(command)
