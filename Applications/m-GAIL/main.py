import sys
import os
from dispatcher import dispatcher

technion=0

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

env_name = 'linemove_2D'
# env_name = 'Roundabout'

sys.path.append(git_path + '/Buffe/utils')
sys.path.append(os.getcwd() + '/environments/' + env_name)

if __name__ == '__main__':

    env = __import__('environment').ENVIRONMENT()

    env.run_dir = git_path + '/Buffe/Applications/m-GAIL/environments/' + env_name + '/'

    dispatcher(env=env)
