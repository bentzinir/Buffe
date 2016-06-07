import sys
from roundabout import ROUNDABOUT

technion=1

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

sys.path.append(git_path + '/Buffe/utils')

from dispatcher import dispatcher

if __name__ == '__main__':

    train_params = {
                    'n_train_iters': 100000,
                    'test_interval': 500,
                    'n_steps_train': 100,
                    'n_steps_test': 150
                     }

    simulator = ROUNDABOUT()

    snapshots_dir = git_path + '/Buffe/Applications/SVG_INF/snapshots/'

    trained_model=''
    # trained_model='/home/nir/work/git/Buffe/Applications/SVG_INF/snapshots/2016-06-07-09-03-016500.sn'

    dispatcher(simulator=simulator, train_params=train_params, trained_model=trained_model, sn_dir=snapshots_dir)