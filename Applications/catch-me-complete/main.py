import os
import sys
sys.path.append('/homes/nirb/work/buffe/utils')
sys.path.append('/homes/nirb/work/buffe/layers')

import theano as t
from optimization import optimization
from io_utils import init_files
import forces

t.config.reoptimize_unpickled_function = False
t.config.scan.allow_output_prealloc = True
t.config.exception_verbosity = 'high'
t.config.warn_float64 = 'warn'

if __name__ == '__main__':

    dt = .1

    game_params = {
                    'dt': dt,
                    'width': 60,
                    'm': 2,
                    'v_max': 5.,
                    'v_target': [1,2],
                    'n_mines': 10,
                    'w_accel': 0.05,
                    'w_progress': 0.5,
                    'w_mines': 0.05,
                    'w_realistic': 0.01,
                    'd_mines': 2,
                    'force': forces.single_sin_force(dt, N=1000, a=[10.,-4.], b=[-8.,6.], T=[1,1.2]),

    }

    controler_0_arch_params = {
        'n_hidden_0': 49,
        'n_steps_train': 15,
        'n_steps_test': 16,
    }

    controler_1_arch_params = {
        'n_hidden_0': 51,
        'n_steps_train': 41,
        'n_steps_test': 26,
    }

    arch_params = {
        'controler_0': controler_0_arch_params,
        'controler_1': controler_1_arch_params,
    }

    # controller 0
    controler_0_solver_params = {
        'lr_type': 'episodic', 'base': 0.01, 'interval': 15e3,
        'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,
        }

    # controller 1
    controler_1_solver_params = {
        'lr_type': 'episodic', 'base': 0.01, 'interval': 15e3,
        'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,
        }

    solver_params = {

                    'controler_0' : controler_0_solver_params,
                    'controler_1' : controler_1_solver_params,

                    # 'lr_type': 'inv', 'base': 0.5, 'gamma': 0.0001, 'power': 0.75,
                    # 'lr_type': 'episodic', 'base': 0.005, 'interval': 10e3,
                    # 'lr_type': 'fixed', 'base': 0.003,
                    # 'optimizer': 'sgd',
                    #'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,
                    #'optimizer': 'rmsprop_graves', 'aleph': 0.95, 'beit': 0.9, 'gimmel': 0.0001, 'dalet': 0.0001,
                    #'optimizer': 'adadelta', 'rho': 0.9, 'eps': 1.E-6,
                    #'optimizer': 'adagrad', 'rho': 0.9, 'eps': 1.E-6,
                    'momentum': 0.9,
                    'n_train_iters': 100000,
                    'test_interval': 1000,
                    'switch_interval': 5000,
                    'grad_clip_val': 10,
                    'l1_weight_decay':0.0001,
                     }

    snapshots_dir = os.getcwd() + '/snapshots/'

    trained_model=['','']
    # trained_model = ''

    optimization(game_params = game_params, arch_params=arch_params, solver_params=solver_params, trained_model= trained_model, sn_dir = snapshots_dir)