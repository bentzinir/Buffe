import sys
import os
import json

from dispatcher import dispatcher

def load_params(fname):
    with open(fname + '/params.json', 'r') as f:
        try:
            data = json.load(f)
        except ValueError:
            data = {}
    return data

def update_env_params(env, p):
    for k, v in p.iteritems():
        setattr(env, k, v)
    return env

# env_name = 'hopper'
env_name = 'walker'

run_dir = '/home/nir/work/git/Buffe/Applications/mgail/environments/' + env_name + '/'

if env_name == 'hopper':
    config_dir = 'models/18T-2016-10-13-10-23-13/config_0/'
    sn = '2016-10-13-12-28-092000.sn'
elif env_name == 'walker':
    config_dir = 'models/18T-2016-10-28-13-11-54/config_7/'
    sn ='2016-10-30-08-09-104000.sn'

sys.path.append(os.getcwd() + '/environments/' + env_name)

env = __import__('environment').ENVIRONMENT(run_dir)

params = load_params(run_dir + config_dir)

env = update_env_params(env, params)

env.train_mode = False

env.trained_model = run_dir + config_dir + sn

dispatcher(env)
