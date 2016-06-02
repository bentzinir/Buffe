import sys
from roundabout import ROUNDABOUT

technion=1

if technion:
    git_path = '/home/nir/work/git/'
else:
    git_path = '/homes/nirb/work/git/'

sys.path.append(git_path + '/Buffe/utils')

from optimization import optimization

if __name__ == '__main__':

    arch_params = {
                    'n_hidden_0': 50,
                    'n_hidden_1': 50,
                    'n_hidden_2': 50,
                    'n_steps_train': 50,
                    'n_steps_test': 50
                   }

    solver_params = {
                    # 'lr_type': 'episodic', 'base': 0.001, 'interval': 5e3,
                    'lr_type': 'inv', 'base': 0.001, 'gamma': 0.0001, 'power': 0.75,
                    #'lr_type': 'fixed', 'base': 0.003,
                    #'optimizer': 'sgd',
                    #'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,
                    #'optimizer': 'rmsprop_graves', 'aleph': 0.95, 'beit': 0.9, 'gimmel': 0.0001, 'dalet': 0.0001,
                    #'optimizer': 'adadelta', 'rho': 0.9, 'eps': 1.E-6,
                    #'optimizer': 'adagrad', 'rho': 0.9, 'eps': 1.E-6,
                    'momentum': 0.9,
                    'n_train_iters': 100000,
                    'test_interval': 500,
                    # 'grad_clip_val': 5,
                    'l1_weight_decay': 0.0001,
                    'weights_stddev': 0.15
                     }

    simulator = ROUNDABOUT()

    snapshots_dir = git_path + '/Buffe/Applications/SVG/snapshots/'

    trained_model=''
    # trained_model = '/homes/nirb/work/buffe/Applications/roundabout-learn/snapshots/2016-01-06-08-22-012000.sn'

    optimization(simulator=simulator, arch_params=arch_params, solver_params=solver_params, trained_model=trained_model, sn_dir=snapshots_dir)