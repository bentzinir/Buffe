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
np.set_printoptions(precision=2)

class CTRL_OPTMZR(object):

    def __init__(self, game_params, arch_params, solver_params, trained_model, sn_dir):

        params=[None, None]

        if trained_model[0]:
            params[0] = common.load_params(trained_model[0])

        if trained_model[1]:
            params[1] = common.load_params(trained_model[1])

        self.lr_func = []
        self.lr_func.append(create_learning_rate_func(solver_params['controler_0']))
        self.lr_func.append(create_learning_rate_func(solver_params['controler_1']))

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
        self.trnsprnt = tt.fscalar('trnsprnt')
        self.rand_goals = tt.fmatrix('rand_goals')
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
                                self.trnsprnt,
                                self.rand_goals,
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
                    self.goal_1,
                    self.trnsprnt,
                    self.rand_goals,
                    ],
            outputs=[self.model.x_h,
                     self.model.x_t,
                     self.model.x_mines,
                     self.model.goal_0,
                     self.model.goal_1,
                     self.model.grad_mean[0],
                     self.model.grad_mean[1],
                     self.model.params_abs_norm[0],
                     self.model.params_abs_norm[1],
                     self.model.c_0_cost_accel,
                     self.model.c_0_cost_progress,
                     self.model.c_1_cost_progress,
                     self.model.c_1_cost_mines,
                     ],
            givens={},
            name='test_model_func',
            on_unused_input='ignore',
        )

        self.train_0_model = t.function(
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
                    self.goal_1,
                    self.trnsprnt,
                    self.rand_goals,
                    ],
            outputs=[self.model.cost[0],
                    ],
            updates=self.model.updates[0],
            givens={},
            name='train_func',
            on_unused_input='ignore',
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
            # mode='DebugMode'
        )

        self.train_1_model = t.function(
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
                    self.goal_1,
                    self.trnsprnt,
                    self.rand_goals,
                    ],
            outputs=[self.model.cost[1],
                    ],
            updates=self.model.updates[1],
            givens={},
            name='train_func',
            on_unused_input='ignore',
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
            # mode='DebugMode'
        )

        self.avg_cost = [999,999]
        self.grad_mean = [0,0]
        self.param_abs_norm = [0,0]
        self.t_switch = 1
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
        rand_goals = (np.random.uniform(low=0, high=60, size=(50,2)).astype(np.float32))

        return x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1, rand_goals

    def train_step(self, itr):

        lr_0 = self.lr_func[0](itr)
        lr_1 = self.lr_func[1](itr)

        x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1, rand_goals = self._drill_points()

        force = self.game_params['force']

        if self.t_switch == 0:
            train_func = self.train_0_model
            arch_params = self.arch_params['controler_0']
            lr = lr_0
        else:
            train_func = self.train_1_model
            arch_params = self.arch_params['controler_1']
            lr = lr_1

        if itr <= self.solver_params['switch_interval']:
            trnsprnt = 1
        else:
            trnsprnt = 0

        returned_train_vals = train_func(x_host_0,
                                        v_host_0,
                                        x_target_0,
                                        v_target_0,
                                        x_mines_0,
                                        time_steps,
                                        force,
                                        arch_params['n_steps_0_train'],
                                        arch_params['n_steps_1_train'],
                                        lr,
                                        goal_1,
                                        trnsprnt,
                                        rand_goals
                                        )

        self.avg_cost[self.t_switch] = 0.95*self.avg_cost[self.t_switch]  + 0.05*returned_train_vals[0]

        if itr % self.solver_params['switch_interval'] == 0:
            self.t_switch = 1 - self.t_switch

        if itr % 100 == 0:
            buf = "processing iter: %d, cost: (%0.2f, %0.2f)" % (itr,self.avg_cost[0],self.avg_cost[1])
            sys.stdout.write('\r'+buf)

    def test_step(self):

        arch_params = self.arch_params['controler_1']

        x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, goal_1, rand_goals = self._drill_points()

        force = self.game_params['force']

        trnsprnt = 0.

        returned_test_vals = self.test_model(x_host_0,
                                             v_host_0,
                                             x_target_0,
                                             v_target_0,
                                             x_mines_0,
                                             time_steps,
                                             force,
                                             arch_params['n_steps_0_test'],
                                             arch_params['n_steps_1_test'],
                                             goal_1,
                                             trnsprnt,
                                             rand_goals
                                            )

        self.x_h_test = returned_test_vals[0]
        self.x_t_test = returned_test_vals[1]
        self.x_mines_test = returned_test_vals[2]
        self.goal_0_test = returned_test_vals[3]
        self.goal_1_test = returned_test_vals[4]
        self.grad_mean[0] = returned_test_vals[5]
        self.grad_mean[1] = returned_test_vals[6]
        self.param_abs_norm[0] = returned_test_vals[7]
        self.param_abs_norm[1] = returned_test_vals[8]

        self.c_0_costs = returned_test_vals[9:11]
        self.c_1_costs = returned_test_vals[11:]

    def play_trajectory(self):

        w = self.game_params['width']
        d_mines = self.game_params['d_mines']

        x_h_3D = self.x_h_test
        x_t_3D = self.x_t_test
        x_mines = self.x_mines_test
        goal_0_2D = self.goal_0_test
        goal_1 = self.goal_1_test

        # loop over trajectory points and plot
        self.ax.clear()
        host_scatter = self.ax.scatter(x_h_3D[0][0][0], x_h_3D[0][0][1], facecolor = 'r', edgecolor='none', s=150)
        target_scatter = self.ax.scatter(x_t_3D[0][0][0], x_t_3D[0][0][1], facecolor = 'y', edgecolor='none', s=30)
        # mines_scatter = self.ax.scatter(x_mines[:,0], x_mines[:,1], facecolor = 'k', edgecolor='none', s=80)
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
                # draw mines
                d_host_mines = ((x_h - x_mines)**2).sum(axis=1)
                activated = d_host_mines <= d_mines
                # mines_scatter.set_array(activated)
                self.ax.scatter(x_mines[:,0], x_mines[:,1], c=activated, s=80)
                self.fig.canvas.draw()
                plt.pause(0.01)
                i += 1

    def print_info_line(self, itr):

        buf = '%s Training ctrlr 1: iter %d, cost %s, grad_mean %s, abs norm: %s, lr %s, c_0_costs: %s, c_1_costs: %s\n' % \
            (datetime.datetime.now(), itr, np.asarray(self.avg_cost), np.asarray(self.grad_mean), np.asarray(self.param_abs_norm),
             np.asarray([f(itr) for f in self.lr_func]), np.asarray(self.c_0_costs), np.asarray(self.c_1_costs))

        sys.stdout.write('\r'+buf)

    def save_model(self, itr):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") +('%0.6d.sn'%itr), obj = common.get_params(self.model.param_struct[0].params + self.model.param_struct[1].params))