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

    env_name = 'Linemove_2d'
    # env_name = 'Roundabout'

    # env = gym.make(env_name)
    env = __import__(env_name).ENVIRONMENT()

    run_dir = git_path + '/Buffe/Applications/m-GAIL/'

    trained_model = None
    # trained_model='/homes/nirb/work/git/Buffe/Applications/m-GAIL/snapshots/2016-06-07-10-53-003500.sn'

    # expert_data = None
    expert_data = run_dir + '/experts/Linemove_2d-2016-08-08-15-31.bin'

    dispatcher(environment=env,
               trained_model=trained_model,
               run_dir=run_dir,
               expert_data=expert_data)