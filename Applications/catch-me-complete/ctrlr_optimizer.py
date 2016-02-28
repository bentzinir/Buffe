import theano as t
import theano.tensor as tt
from learning_rate import create_learning_rate_func
from theano.compile.nanguardmode import NanGuardMode
from controller import CONTROLLER
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import datetime
import time
import common
import sys

class CTRL_OPTMZR(object):

    def __init__(self, game_params, arch_params, solver_params, trained_model, sn_dir):

        params=[None, None]

        if trained_model[0]:
            params[0] = common.load_params(trained_model[0])

        if trained_model[1]:
            params[1] = common.load_params(trained_model[1])

        self.lr_func = create_learning_rate_func(solver_params)

        self.x_host_0 = tt.fvector('x_host_0')
        self.v_host_0 = tt.fvector('v_host_0')
        self.x_target_0 = tt.fvector('x_target_0')
        self.v_target_0 = tt.fvector('v_target_0')
        self.x_mines_0 = tt.fmatrix('x_mines_0')
        self.time_steps = tt.fvector('time_steps')
        self.force = tt.fmatrix('force')
        self.n_steps_0 = tt.iscalar('n_steps_0')
        self.n_steps_1 = tt.iscalar('n_steps_1')
        self.lr = tt.fscalar('lr')
        self.goal_1 = tt.fvector('goal_1')
        self.game_params = game_params
        self.arch_params = arch_params
        self.solver_params = solver_params
        self.sn_dir = sn_dir

        self.model = CONTROLLER(self.x_host_0,
                                self.v_host_0,
                                self.x_target_0,
                                self.v_target_0,
                                self.x_mines_0,
                                self.time_steps,
                                self.force,
                                self.n_steps_0,
                                self.n_steps_1,
                                self.lr,
                                self.goal_1,
                                self.game_params,
                                self.arch_params,
                                self.solver_params,
                                params)

    def init_theano_funcs(self):

        self.test_model = t.function(
            inputs=[self.x_host_0,
                    self.v_host_0,
                    self.x_target_0,
                    self.v_target_0,
                    self.x_mines_0,
                    self.time_steps,
                    self.force,
                    self.n_steps_0,
                    self.n_steps_1,
                    self.goal_1
                    ],
            outputs=[self.model.x_h,
                     self.model.x_t,
                     self.model.x_mines,
                     self.model.goal_0,
                     self.model.goal_1,
                     self.model.grad_mean[0],
                     self.model.grad_mean[1],
                     self.model.params_abs_norm[0],
                     self.model.params_abs_norm[1]
                     ],
            givens={},
            name='test_model_func',
            on_unused_input='ignore',
        )

        self.train_model = t.function(
            inputs=[self.x_host_0,
                    self.v_host_0,
                    self.x_target_0,
                    self.v_target_0,
                    self.x_mines_0,
                    self.time_steps,
                    self.force,
                    self.n_steps_0,
                    self.n_steps_1,
                    self.lr,
                    self.goal_1
                    ],
            outputs=[self.model.cost[0],
                     self.model.cost[1]
                    ],
            updates=self.model.updates,
            givens={},
            name='train_func',
            on_unused_input='ignore',
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
            # mode='DebugMode'
        )

        self.avg_cost = [999,999]
        self.grad_mean = [0,0]
        self.param_abs_norm = [0,0]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        plt.ion()
        plt.show()

    def _drill_points(self):
        w = self.game_params['width']
        v_max = self.game_params['v_max']
        n_mines = self.game_params['n_mines']

        x_host_0 = np.float32(np.random.uniform(low=0, high=w, size=2))
        v_host_0 = np.float32(np.random.uniform(low=-v_max, high=v_max, size=2))
        x_target_0 = np.float32(np.random.uniform(low=0, high=w, size=2))
        v_target_0 = np.asarray(self.game_params['v_target']).astype(np.float32)
        x_mines_0 = np.random.uniform(low=0, high=w, size=(n_mines,2)).astype(np.float32)
        time_steps = np.asarray(xrange(1000)).astype(np.float32)
        goal_1 = (np.random.uniform(low=0, high=60, size=2).astype(np.float32))

        return x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1

    def train_step(self, iter):
        n_steps_0 = self.arch_params['n_steps_0_train']
        n_steps_1 = self.arch_params['n_steps_1_train']

        x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1 = self._drill_points()

        force = self.game_params['force']

        returned_train_vals = self.train_model(x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, force, n_steps_0, n_steps_1, self.lr_func(iter), goal_1)
        self.avg_cost[0] = 0.95*self.avg_cost[0]  + 0.05*returned_train_vals[0]
        self.avg_cost[1] = 0.95*self.avg_cost[1]  + 0.05*returned_train_vals[1]

        if iter % 100 == 0:
            buf = "processing iter: %d" % iter
            sys.stdout.write('\r'+buf)

    def test_step(self):
        n_steps_0 = self.arch_params['n_steps_0_test']
        n_steps_1 = self.arch_params['n_steps_1_test']

        x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1 = self._drill_points()

        force = self.game_params['force']

        returned_test_vals = self.test_model(x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, force, n_steps_0, n_steps_1, goal_1)

        self.x_h_test = returned_test_vals[0]
        self.x_t_test = returned_test_vals[1]
        self.x_mines_test = returned_test_vals[2]
        self.goal_0_test = returned_test_vals[3]
        self.goal_1_test = returned_test_vals[4]
        self.grad_mean[0] = returned_test_vals[5]
        self.grad_mean[1] = returned_test_vals[6]
        self.param_abs_norm[0] = returned_test_vals[7]
        self.param_abs_norm[1] = returned_test_vals[8]

    def play_trajectory(self):

        w = self.game_params['width']

        x_h_3D = self.x_h_test
        x_t_3D = self.x_t_test
        x_mines = self.x_mines_test
        goal_0_2D = self.goal_0_test
        goal_1 = self.goal_1_test

        # loop over trajectory points and plot
        self.ax.clear()
        host_scatter = self.ax.scatter(x_h_3D[0][0][0], x_h_3D[0][0][1], facecolor = 'r', edgecolor='none', s=150)
        target_scatter = self.ax.scatter(x_t_3D[0][0][0], x_t_3D[0][0][1], facecolor = 'y', edgecolor='none', s=30)
        mines_scatter = self.ax.scatter(x_mines[:,0], x_mines[:,1], facecolor = 'k', edgecolor='none', s=80)
        goal_0_scatter = self.ax.scatter(goal_0_2D[0][0], goal_0_2D[0][1], marker='x', s=80, color='b')
        goal_1_scatter = self.ax.scatter(goal_1[0], goal_1[1], marker='x', s=80, color='r')
        self.fig.canvas.draw()

        self.ax.add_patch(Rectangle((0, 0), w, w, alpha=0.1, edgecolor='k', facecolor='b'))
        self.ax.set_xlim([-10, w+10])
        self.ax.set_ylim([-10, w+10])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        i = 0
        for x_h_2D, x_t_2D, goal_0 in zip(x_h_3D, x_t_3D, goal_0_2D):

            goal_0_scatter.set_offsets(goal_0)

            for x_h, x_t in zip (x_h_2D, x_t_2D):
                self.ax.set_title(('Sample Trajectory\n time: %d' % i))
                host_scatter.set_offsets(x_h)
                target_scatter.set_offsets(x_t)
                self.fig.canvas.draw()
                time.sleep(0.01)
                i += 1

    def print_info_line(self, iter):

        buf = '%s Training ctrlr 1: iter %d, cost (%0.2f, %0.2f), grad_mean (%0.2f, %0.2f), abs norm: (%0.2f, %0.2f), lr %0.6f\n' % \
                                    (datetime.datetime.now(), iter, self.avg_cost[0], self.avg_cost[1],
                                     self.grad_mean[0], self.grad_mean[1],
                                     self.param_abs_norm[0], self.param_abs_norm[0], self.lr_func(iter))
        sys.stdout.write('\r'+buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") +('%0.6d.sn'%iter), obj = common.get_params(self.model.param_struct[0].params + self.model.param_struct[1].params))