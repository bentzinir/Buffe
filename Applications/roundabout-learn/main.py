
import sys
sys.path.append('/homes/nirb/work/buffe/utils')
sys.path.append('/homes/nirb/work/buffe/layers')

import theano as t
from optimization import optimization
from io_utils import init_files

t.config.reoptimize_unpickled_function = False
t.config.scan.allow_output_prealloc = True
t.config.exception_verbosity = 'high'
t.config.warn_float64 = 'warn'

if __name__ == '__main__':

    game_params = {
                    'r': 10.,
                    'dt': 0.1,
                    'v_0': 5.,
                    'num_of_targets': 5,
                    'dist_in_seconds': 1.8,
                    'alpha_accel': 0.01,
                    'alpha_progress': 0.1,
                    'alpha_accident': 5,
                    'require_distance': 5,
                    'p_aggressive': 0.5,
                    'host_length': 5,
                    'alpha_cost_a': 0.5,
                    'alpha_cost_t': 0.99,
    }

    arch_params = {
                    'n_hidden_a_0': 100,
                    'n_hidden_a_1': 100,
                    'n_hidden_a_2': 50,
                    'n_hidden_t_0': 100,
                    'n_hidden_t_1': 100,
                    'n_hidden_t_2': 50,

                    'n_steps_train': 300,
                    'n_steps_test': 300
                   }

    solver_params = {
                    # controller 0
                    # 'lr_type': 'episodic', 'base': 0.005, 'interval': 10e3,
                    # 'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,

                    # controller 1
                    'lr_type': 'episodic', 'base': 0.0005, 'interval': 5e3,
                    'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,

                    #'lr_type': 'inv', 'base': 0.5, 'gamma': 0.0001, 'power': 0.75,
                    #'lr_type': 'episodic', 'base': 0.005, 'interval': 10e3,
                    #'lr_type': 'fixed', 'base': 0.003,
                    #'optimizer': 'sgd',
                    #'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,
                    #'optimizer': 'rmsprop_graves', 'aleph': 0.95, 'beit': 0.9, 'gimmel': 0.0001, 'dalet': 0.0001,
                    #'optimizer': 'adadelta', 'rho': 0.9, 'eps': 1.E-6,
                    #'optimizer': 'adagrad', 'rho': 0.9, 'eps': 1.E-6,
                    'momentum': 0.9,
                    'n_train_iters': 100000,
                    'test_interval': 1000,
                    # 'grad_clip_val': 5,
                    'l1_weight_decay':0.0001,
                     }

    snapshots_dir = '/homes/nirb/work/buffe/Applications/roundabout-learn/snapshots/'

    trained_model=''
    # trained_model = '/homes/nirb/work/buffe/Applications/roundabout-learn/snapshots/2016-01-06-08-22-012000.sn'

    optimization(game_params = game_params, arch_params=arch_params, solver_params=solver_params, trained_model= trained_model, sn_dir = snapshots_dir)