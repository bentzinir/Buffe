import sys
import os
import time
import random
import json

from dispatcher import dispatcher

env_name = 'halfcheetah'
n_trajs = '25'

sys.path.append(os.getcwd() + '/environments/' + env_name)

run_dir = '/home/nir/work/git/Buffe/Applications/mgail/environments/' + env_name + '/'

env = __import__('environment').ENVIRONMENT(run_dir)

env.save_models = False
env.vis_flag = False
env.n_train_iters = 120001
env.n_episodes_test = 20
env.train_mode = True
env.expert_data = 'expert_data/expert-2016-11-06-11-07.bin'
env.kill_itr = 50000
env.reward_kill_th = 500
env.success_th = 4000

# grid search parameters
params = {
    'collect_experience_interval': [5, 15],
    'n_steps_train': [25, 50, 75, 100],
    'discr_policy_itrvl': [100, 250, 500],
    'K_T': [1, 2, 3, 4],
    'K_D': [1, 2],
    'K_P': [1, 2, 3, 4],
    'do_keep_prob': [0.6, 0.75, 0.8],
    'policy_al_w': [1e-1, 1e-0],
    'batch_size': [50, 70, 100],
    'er_agent_size': [50000, 100000, 150000],
    'p_size': [[75, 25], [100, 50], [150, 75]],
    'd_size': [[50, 25], [100, 50], [150, 75], [200, 100]],
    't_size': [[200, 100], [400, 200], [800, 400]],
    'p_lr': [0.002, 0.001, 0.0001],
    'd_lr': [0.002, 0.001, 0.005],
    't_lr': [0.001, 0.0005, 0.0001, 0.00001],
    'noise_intensity': [3., 4., 5., 6.],
}

def drill_params(p):
    p_random = dict.copy(p)
    for k, v in p.iteritems():
        p_random[k] = random.choice(v)
    return p_random

def update_env_params(env, p):
    for k, v in p.iteritems():
        # if hasattr(env, k):
        setattr(env,k,v)
    return env

def load_params(fname):
    with open(fname + '/params.json', 'r') as f:
        try:
            data = json.load(f)
        # if the file is empty the ValueError will be thrown
        except ValueError:
            data = {}
    return data

def save_params(obj, fname):
    with open(fname + '/params.json', 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=False)

time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists(run_dir + n_trajs + 'T-' + time_str + '/'):
    os.makedirs(run_dir + n_trajs + 'T-' + time_str + '/')

for i in xrange(1000):

    config_dir = run_dir + n_trajs + 'T-' + time_str + '/' + '/config_' + str(i) + '/'

    # create run dir
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    with open(config_dir + 'log_file.txt', 'wb') as f:
        f.write('Log File...\n\n\n')

        params_i = drill_params(params)

        env = update_env_params(env, params_i)

        env.config_dir = config_dir

        # save params to config_dir
        save_params(params_i, config_dir)

        # run job
        env.log_fid = f
        dispatcher(env)
