import sys
import gym
import os

technion=1

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

sys.path.append(git_path + '/Buffe/utils')
sys.path.append(os.getcwd() + '/configurations')

from dispatcher import dispatcher

if __name__ == '__main__':

    # env_name = 'InvertedPendulum-v1'
    # env_name = 'InvertedDoublePendulum-v1'
    # env_name = 'Reacher-v1'
    env_name = 'BipedalWalker-v2'
    env = gym.make(env_name)

    config = __import__(env_name).CONFIGURATION()

    snapshots_dir = git_path + '/Buffe/Applications/SVG_1/snapshots/'

    trained_model = ''
    # trained_model='/homes/nirb/work/git/Buffe/Applications/SVG_INF/snapshots/2016-06-07-10-53-003500.sn'

    dispatcher(environment=env, configuration=config, trained_model=trained_model, sn_dir=snapshots_dir)