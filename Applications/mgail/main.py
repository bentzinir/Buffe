import sys
import os

technion=1

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

sys.path.append(git_path + '/Buffe/utils')
sys.path.append(git_path + '/Buffe/Applications/mgail/')

from dispatcher import dispatcher

# env_name = 'linemove_2D'
# env_name = 'roundabout'
# env_name = 'hopper'
# env_name = 'walker'
# env_name = 'halfcheetah'
#  env_name = 'ant'
env_name = 'mountaincar'

sys.path.append(os.getcwd() + '/environments/' + env_name)

if __name__ == '__main__':

    run_dir = git_path + '/Buffe/Applications/mgail/environments/' + env_name + '/'
    env = __import__('environment').ENVIRONMENT(run_dir)

    dispatcher(env=env)
