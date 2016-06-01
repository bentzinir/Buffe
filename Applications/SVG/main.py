import sys
sys.path.append('/homes/nirb/work/git/Buffe/utils')

from optimization import optimization

if __name__ == '__main__':

    game_params = {
                    'r': 10.,
                    'dt': 0.1,
                    'v_0': 5.,
                    'num_of_targets': 5,
                    'dist_in_seconds': 1.8,
                    'alpha_accel': 0.,
                    'alpha_progress': 0.9,
                    'alpha_accident': 5,
                    'require_distance': 5,
                    'p_aggressive': 0.5,
                    'host_length': 5,
                    'alpha_cost_a': 0.5,
                    'alpha_cost_t': 0.99,
                    'gamma':0.95,
    }

    arch_params = {
                    'n_hidden_0': 50,
                    'n_hidden_1': 50,
                    'n_hidden_2': 25,
                    'n_steps_train': 100,
                    'n_steps_test': 100
                   }

    solver_params = {
                    'lr_type': 'episodic', 'base': 0.05, 'interval': 5e3,
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
                    'test_interval': 50,
                    # 'grad_clip_val': 5,
                    'l1_weight_decay':0.0001,
                     }

    snapshots_dir = '/homes/nirb/work/git/Buffe/Applications/SVG/snapshots/'

    trained_model=''
    # trained_model = '/homes/nirb/work/buffe/Applications/roundabout-learn/snapshots/2016-01-06-08-22-012000.sn'

    optimization(game_params = game_params, arch_params=arch_params, solver_params=solver_params, trained_model= trained_model, sn_dir = snapshots_dir)