import os
import sys
sys.path.append('../../utils')
sys.path.append('../../layers')

import theano as t
from optimization import optimization
from io_utils import init_files

t.config.reoptimize_unpickled_function = False
t.config.scan.allow_output_prealloc = True
t.config.exception_verbosity = 'high'
t.config.warn_float64 = 'warn'

if __name__ == '__main__':

    game_params = {
                    'dt': 0.1,
                    'v0': 1.,
                    'v_max': 1.,
                    'board_width': 1,
                    'lane_width':0.3,
                    'p_exist': 0.6,
                    'p_leader': 0.4,
                    't_leader_wait': 3,

                    'w_delta_steer': 0.1,
                    'w_accel': 0.1,
                    'w_progress': 0.2,
                    'w_accident': 0.1,
                    'w_row': 0,
                    'w_rail': 0.1,

                    'require_distance': 0.1,
                    'p_aggressive': 0.5,
                    'host_length': 0.1,
                    'eps_row': 0.1,
    }

    arch_params = {
                    'n_hidden_0': 100,
                    'n_hidden_1': 75,
                    'n_hidden_2': 50,
                    'n_steps_train': 200,
                    'n_steps_test': 200
                   }

    solver_params = {

                    'lr_type': 'episodic', 'base': 0.005, 'interval': 1e4,
                    'optimizer': 'rmsprop', 'rho': 0.9, 'eps': 1E-6,

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
                    'test_interval': 200,
                    'grad_clip_val': 10,
                    'l1_weight_decay':0.00001,
                     }

    snapshots_dir = os.getcwd() + '/snapshots/'

    trained_model=''
    # trained_model = '/homes/nirb/work/buffe/Applications/4-way-stop/snapshots/2016-02-18-12-58-063000.sn'

    optimization(game_params = game_params, arch_params=arch_params, solver_params=solver_params, trained_model= trained_model, sn_dir = snapshots_dir)