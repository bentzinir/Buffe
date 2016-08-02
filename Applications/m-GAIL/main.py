import sys
import os
from dispatcher import dispatcher

technion=1

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

sys.path.append(git_path + '/Buffe/utils')
sys.path.append(os.getcwd() + '/configurations')
sys.path.append(os.getcwd() + '/environments')

if __name__ == '__main__':

    env_name = 'Roundabout'

    # env = gym.make(env_name)
    env = __import__(env_name).ENVIRONMENT()

    config = __import__(env_name+'-config').CONFIGURATION()

    snapshots_dir = git_path + '/Buffe/Applications/m-GAIL/snapshots/'

    trained_model = ''
    # trained_model='/homes/nirb/work/git/Buffe/Applications/m-GAIL/snapshots/2016-06-07-10-53-003500.sn'

    expert_data = ''

    dispatcher(environment=env,
               configuration=config,
               trained_model=trained_model,
               sn_dir=snapshots_dir,
               expert_data=expert_data)