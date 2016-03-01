import theano as t
import theano.tensor as tt
from learning_rate import create_learning_rate_func
from controller import CONTROLLER
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import numpy as np
import datetime
import time
import common

class CTRL_OPTMZR(object):

    def __init__(self, game_params, arch_params, solver_params, trained_model, sn_dir):

        params=None

        if trained_model:
            params = common.load_params(trained_model)

        self.lr_func = create_learning_rate_func(solver_params)
        self.v_h_0 = tt.fvector('v_h_0')
        self.x_h_0 = tt.fvector('x_h_0')
        self.v_t_0 = tt.fmatrix('v_t_0')
        self.x_t_0 = tt.fmatrix('x_t_0')
        self.a_t_0 = tt.fmatrix('a_t_0')
        self.is_aggressive = tt.fmatrix('is_aggressive')
        self.lr_ = tt.fscalar('lr')
        self.n_steps_ = tt.iscalar('n_steps')
        self.sn_dir = sn_dir
        self.game_params = game_params
        self.arch_params = arch_params
        self.solver_params = solver_params

        self.model = CONTROLLER(self.v_h_0, self.x_h_0, self.v_t_0, self.x_t_0, self.a_t_0, self.is_aggressive, self.lr_, self.n_steps_, self.game_params, self.arch_params, self.solver_params, params)

    def init_theano_funcs(self):

        self.test_model = t.function(
            inputs=[self.v_h_0, self.x_h_0, self.v_t_0, self.x_t_0, self.a_t_0, self.is_aggressive, self.n_steps_],
            outputs=[self.model.v_h, self.model.x_h, self.model.v_t, self.model.x_t, self.model.grad_mean, self.model.params_abs_norm],
            givens={},
            name='test_model_func',
            on_unused_input='ignore',
        )

        self.train_model = t.function(
            inputs=[self.v_h_0, self.x_h_0, self.v_t_0, self.x_t_0, self.a_t_0, self.is_aggressive, self.lr_, self.n_steps_],
            outputs=[self.model.cost_a, self.model.cost_t],
            updates=self.model.updates,
            givens={},
            name='train_func',
            on_unused_input='ignore',
        )

        self.avg_cost_a = 999
        self.avg_cost_t = 999
        self.grad_mean = 0
        self.param_abs_norm = 0

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=2)

        plt.ion()
        plt.show()
        plt.axis('equal')

    def _drill_points(self):
        r = self.game_params['r']
        n_t = self.game_params['num_of_targets']
        v_h_0 = np.asarray([self.game_params['v_0']]).astype(np.float32)
        x_h_0 = - 3 * np.asarray([r]).astype(np.float32)
        v_t_0 = self.game_params['v_0'] * np.ones(shape=(n_t,1)).astype(np.float32)
        x_t_0 = np.sort(np.random.uniform(low= -r*np.pi, high = r*np.pi, size=n_t)).reshape(n_t,1).astype(np.float32)
        a_t_0 = np.zeros_like(v_t_0).astype(np.float32)
        is_aggressive = np.random.binomial(n=1, p=self.game_params['p_aggressive'], size=(n_t,1)).astype(np.float32)

        return v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive

    def train_step(self, iter):
        n_steps = self.arch_params['n_steps_train']
        v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive = self._drill_points()

        returned_train_vals = self.train_model(v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive, self.lr_func(iter), n_steps)
        cost_a = returned_train_vals[0]
        cost_t = returned_train_vals[1]
        self.avg_cost_a = 0.95*self.avg_cost_a  + 0.05*cost_a
        self.avg_cost_t = 0.95*self.avg_cost_t  + 0.05*cost_t

    def test_step(self):
        n_steps = self.arch_params['n_steps_test']
        v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive = self._drill_points()

        returned_test_vals = self.test_model(v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive, n_steps)

        self.v_h_test = returned_test_vals[0]
        self.x_h_test = returned_test_vals[1]
        self.v_t_test = returned_test_vals[2]
        self.x_t_test = returned_test_vals[3]
        self.grad_mean = returned_test_vals[4]
        self.param_abs_norm = returned_test_vals[5]

        self.is_aggressive_test = is_aggressive

    def play_trajectory(self):

        r = self.game_params['r']
        x_h_test = self.x_h_test
        v_h_test = self.v_h_test
        x_t_test = self.x_t_test
        v_t_test = self.v_t_test

        is_aggressive = self.is_aggressive_test

        self.ax.set_title('Sample Trajectory')

        x_h_world = []
        v_h_world = []
        for x_h, v_h in zip(x_h_test, v_h_test):
            x_h = x_h[0]
            x,y = (x_h > 0)*r*np.array([np.cos(x_h/r-np.pi/2),np.sin(x_h/r-np.pi/2)]) + (x_h <= 0)*np.array([0,-r+x_h])
            x_h_world.append([x,y])
            v_h_world.append(v_h[0])

        x_t_world = []
        v_t_world = []
        for x_t, v_t in zip(x_t_test, v_t_test):
            x_t = x_t.transpose()[0]
            x = r*np.cos(x_t/r - np.pi/2)
            y = r*np.sin(x_t/r - np.pi/2)
            x_t_world.append([x,y])
            v_t_world.append(v_t.transpose()[0])

        # loop over trajectory points and plot
        for x_h, v_h, x_t, v_t in zip(x_h_world, v_h_world, x_t_world, v_t_world):
            self.ax.clear()
            circle = plt.Circle((0,0), radius = self.game_params['r'], edgecolor='k', facecolor='none', linestyle='dashed')
            self.fig.gca().add_artist(circle)

            # self.ax.add_patch(Rectangle((x_h[0],x_h[1]),2,2))
            self.ax.scatter(x_h[0],x_h[1], s=180, marker='o', color='r')
            self.ax.annotate(v_h, (x_h[0],x_h[1]))

            for tx,ty, v_, is_ag in zip(x_t[0], x_t[1], v_t, is_aggressive):
                if is_ag:
                    self.ax.scatter(tx,ty, s=180, marker='o', color='b')
                    self.ax.annotate(v_, (tx,ty))
                else:
                    self.ax.scatter(tx,ty, s=180, marker='o', color='g')
                    self.ax.annotate(v_, (tx,ty))

            self.ax.set_xlim([-2*self.game_params['r'], 2*self.game_params['r']])
            self.ax.set_ylim([-3*self.game_params['r'], 2*self.game_params['r']])
            self.fig.canvas.draw()
            time.sleep(0.03)

    def print_info_line(self, iter):

        buf = '%s Training ctrlr 1: iter %d, cost_a: %0.4f, cost_t: %0.4f, grad_mean %0.6f, abs norm %0.2f, lr %0.6f' % \
                                    (datetime.datetime.now(), iter, self.avg_cost_a, self.avg_cost_t, self.grad_mean, self.param_abs_norm, self.lr_func(iter))
        print(buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") +('%0.6d.sn'%iter), obj = common.get_params(self.model.params))