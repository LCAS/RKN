import os

"""Launches jobs on the cluster (ATTENTION: The number of jobs grows quite fast)
However, this is probably useless for other people anyway since it relies on my modified version of ezex... 
... write your own scripts ;)"""

"""Configure"""

task_modes = ["filter"]
output_modes = ["positions"]
transition_models = ["rkn", "gru", "lstm"]
trials = 3

"""run"""

for transition_model in transition_models:
    for i in range(trials):
       command = ("ezex run -nvd -tl 'rkn/runQuadLinkExperiments.py" +
                  " --transition_model " + transition_model +
                  "' ql_filter_" + transition_model + '{:0>2}'.format(i + 1))
       print(command)
              #  os.system(command)


