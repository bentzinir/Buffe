import theano as t
import theano.tensor as tt
from learning_rate import create_learning_rate_func
from theano.compile.nanguardmode import NanGuardMode
from controller import CONTROLLER
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, Ellipse
from numpy import linalg as LA
import numpy as np
import datetime
import time
import common
import sys

class CTRL_OPTMZR(object):

    def __init__(self, game_params, arch_params, solver_params, trained_model, sn_dir):

        params=None

        if trained_model:
            params = common.load_params(trained_model)

        self.lr_func = create_learning_rate_func(solver_params)

        self.x_h_0 = tt.fvector('x_h_0')
        self.v_h_0 = tt.fvector('v_h_0')
        self.t_h_0 = tt.fvector('t_h_0')
        self.x_t_0 = tt.fmatrix('x_t_0')
        self.v_t_0 = tt.fmatrix('v_t_0')
        self.a_t_0 = tt.fmatrix('a_t_0')
        self.t_t_0 = tt.fvector('t_t_0')
        self.time_steps = tt.fvector('t_0')
        self.exist = tt.bvector('exist')
        self.is_leader = tt.fvector('is_leader')
        self.x_goal = tt.fvector('x_goal')
        self.turn_vec_h = tt.fvector('turn_vec_h')
        self.turn_vec_t = tt.fvector('turn_vec_t')
        self.n_steps = tt.iscalar('n_steps')
        self.lr = tt.fscalar('lr')
        self.sn_dir = sn_dir
        self.game_params = game_params
        self.arch_params = arch_params
        self.solver_params = solver_params

        self.model = CONTROLLER(self.x_h_0,
                                self.v_h_0,
                                self.t_h_0,
                                self.x_t_0,
                                self.v_t_0,
                                self.a_t_0,
                                self.t_t_0,
                                self.time_steps,
                                self.exist,
                                self.is_leader,
                                self.x_goal,
                                self.turn_vec_h,
                                self.turn_vec_t,
                                self.n_steps,
                                self.lr,
                                self.game_params,
                                self.arch_params,
                                self.solver_params,
                                params)

    def init_theano_funcs(self):

        self.test_model = t.function(
            inputs=[self.x_h_0,
                    self.v_h_0,
                    self.t_h_0,
                    self.x_t_0,
                    self.v_t_0,
                    self.a_t_0,
                    self.t_t_0,
                    self.time_steps,
                    self.exist,
                    self.is_leader,
                    self.x_goal,
                    self.turn_vec_h,
                    self.turn_vec_t,
                    self.n_steps
            ],
            outputs=[self.model.x_h,
                     self.model.v_h,
                     self.model.x_t,
                     self.model.v_t,
                     self.model.grad_mean,
                     self.model.params_abs_norm,
                     self.model.max_grad_val
            ],
            givens={},
            name='test_model_func',
            on_unused_input='ignore',
        )

        self.train_model = t.function(
            inputs=[self.x_h_0,
                    self.v_h_0,
                    self.t_h_0,
                    self.x_t_0,
                    self.v_t_0,
                    self.a_t_0,
                    self.t_t_0,
                    self.time_steps,
                    self.exist,
                    self.is_leader,
                    self.x_goal,
                    self.turn_vec_h,
                    self.turn_vec_t,
                    self.n_steps,
                    self.lr
            ],
            outputs=[self.model.cost,
                     self.model.cost_accel,
                     self.model.cost_progress,
                     self.model.cost_rail,
                     self.model.cost_accident,
                     self.model.cost_row,
                     self.model.cost_steer
                     ],
            updates=self.model.updates,
            givens={},
            name='train_func',
            on_unused_input='ignore',
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
            # mode='DebugMode'
        )

        self.avg_cost = 999

        self.avg_cost_progress = 999
        self.avg_cost_rail = 999
        self.avg_cost_accident = 999
        self.avg_cost_accel = 999
        self.avg_cost_row = 999
        self.avg_cost_steer = 999

        self.grad_mean = 0
        self.param_abs_norm = 0
        self.max_grad = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        plt.ion()
        plt.show()
        plt.axis('equal')

    def _drill_points(self):
        bw = self.game_params['board_width']
        lw = self.game_params['lane_width']
        p_exist = self.game_params['p_exist']
        p_leader = self.game_params['p_leader']
        v0 = self.game_params['v0']
        t_ldr_wait = self.game_params['t_leader_wait']

        origin_mat = np.asarray([[bw,lw/2],[-lw/2,bw],[-bw,-lw/2]])
        direction_mat = np.asarray([[-1,0],[0,-1],[1,0]])

        # 1. "exist" random variable. if 0: delete car in lane
        exist = (np.random.binomial(n=1, p=p_exist, size=(3,1))).astype(np.bool)

        # 2. offsets:
        offsets = exist * np.random.uniform(low=0,high=bw-lw,size=((3,1)))

        # 3. drill "leader". if 1: promote the front car to the stopping line
        leader = np.random.binomial(n=1, p=p_leader, size=1)

        is_leader = np.zeros((3),dtype=np.float32)
        if leader:
            j = np.argmax(offsets)
            offsets[j] = bw-lw
            is_leader[j] = 1

        x_t_0 = (origin_mat + direction_mat * offsets).astype(np.float32)

        # 4. assign speeds (linear function with 0 at the stopping line)
        # v_abs = exist * (v0 -(v0/(bs-lw)) * (offsets))
        v_abs = exist * 0.5
        v_t_0 = (v_abs * direction_mat).astype(np.float32)

        # 5. assign accelerations ( can be derived from the speed function and dt)
        a_t_0 = (-v0/(bw-lw) * np.ones((3,1)) * direction_mat).astype(np.float32)
        a_t_0 = 0 * v_t_0

        # 6. assign waiting times
        t_t_0 = np.zeros((3), dtype=np.float32)

        if leader:
            t_t_0[j] = np.clip(np.random.normal(loc=t_ldr_wait, scale=1, size=1), 1.5, 5)

        # 7. host turn type
        x_goal_mat = np.asarray([[lw,-lw/2],[lw/2,lw],[-lw,lw/2]],dtype=np.float32) # right, front, left
        turn_type = np.random.randint(low=0, high=3)
        turn_vec_h = np.zeros(3,dtype=np.float32)
        turn_vec_h[turn_type] = 1
        x_goal = x_goal_mat[turn_type]

        # 7.1 target turn types
        turn_vec_t = np.random.randint(low=0, high=3, size=3).astype(np.float32)

        # 8. initialize host car at the bottom lane
        x_h_0 = np.asarray([lw/2, -lw]).astype(np.float32)
        v_h_0 = np.zeros_like(x_h_0).astype(np.float32)
        t_h_0 = np.zeros((1),dtype=np.float32)

        time_steps = np.asarray(xrange(1000)).astype(np.float32)

        return x_h_0, v_h_0, t_h_0, x_t_0, v_t_0, a_t_0, t_t_0, time_steps, np.squeeze(exist), is_leader, x_goal, turn_vec_h, turn_vec_t

    def train_step(self, iter):
        n_steps = self.arch_params['n_steps_train']
        x_h_0, v_h_0, t_h_0, x_t_0, v_t_0, a_t_0, t_t_0, time_steps, exist, is_leader, x_goal, turn_vec_h, turn_vec_t = self._drill_points()

        returned_train_vals = self.train_model(x_h_0, v_h_0, t_h_0, x_t_0, v_t_0, a_t_0, t_t_0, time_steps, exist, is_leader, x_goal, turn_vec_h, turn_vec_t, n_steps, self.lr_func(iter))
        cost = returned_train_vals[0]

        self.avg_cost = 0.95*self.avg_cost  + 0.05*cost

        self.avg_cost_progress = 0.95*self.avg_cost_progress  + 0.05*returned_train_vals[1]
        self.avg_cost_rail = 0.95*self.avg_cost_rail  + 0.05*returned_train_vals[2]
        self.avg_cost_accident = 0.95*self.avg_cost_accident  + 0.05*returned_train_vals[3]
        self.avg_cost_accel = 0.95*self.avg_cost_accel  + 0.05*returned_train_vals[4]
        self.avg_cost_row = 0.95*self.avg_cost_row  + 0.05*returned_train_vals[5]
        self.avg_cost_steer = 0.95*self.avg_cost_steer  + 0.05*returned_train_vals[6]

        if iter % 100 == 0:
            buf = "processing iter: %d" % iter
            sys.stdout.write('\r'+buf)

    def test_step(self):
        n_steps = self.arch_params['n_steps_test']
        x_h_0, v_h_0, t_h_0, x_t_0, v_t_0, a_t_0, t_t_0, time_steps, exist, is_leader, x_goal, turn_vec_h, turn_vec_t = self._drill_points()

        returned_test_vals = self.test_model(x_h_0, v_h_0, t_h_0, x_t_0, v_t_0, a_t_0, t_t_0, time_steps, exist, is_leader, x_goal, turn_vec_h, turn_vec_t, n_steps)

        self.x_h_test = returned_test_vals[0]
        self.v_h_test = returned_test_vals[1]
        self.x_t_test = returned_test_vals[2]
        self.v_t_test = returned_test_vals[3]
        self.grad_mean = returned_test_vals[4]
        self.param_abs_norm = returned_test_vals[5]
        self.max_grad = returned_test_vals[6]

        self.exist_test = exist
        self.is_leader_test = is_leader
        self.turn_vec_test = turn_vec_h

    def play_trajectory(self):

        bw = self.game_params['board_width']
        lw = self.game_params['lane_width']

        x_hs = self.x_h_test
        v_hs = self.v_h_test
        x_ts = self.x_t_test
        v_ts = self.v_t_test
        exist_vec = self.exist_test
        leader_vec = self.is_leader_test
        turn_vec = self.turn_vec_test

        switcher ={
            0: "right",
            1: "straight",
            2: "left"}

        turn_name = switcher.get(np.nonzero(turn_vec)[0][0])

        i = 0
        # loop over trajectory points and plot
        for x_h, v_h, x_t, v_t in zip(x_hs, v_hs, x_ts, v_ts):
            self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n Turn type: %s,   time: %d' % (turn_name, i)))
            self._print_junction()

            self.ax.scatter(x_h[0],x_h[1], s=180, marker='o', color='r')
            self.ax.annotate(LA.norm(v_h), (x_h[0],x_h[1]))

            for t, v_, exist, lead in zip(x_t, v_t, exist_vec, leader_vec):
                if exist:
                    if lead:
                        self.ax.scatter(t[0],t[1], s=180, marker='o', color='b')
                    else:
                        self.ax.scatter(t[0],t[1], s=180, marker='o', color='g')
                    self.ax.annotate(LA.norm(v_), (t[0],t[1]))

            self.ax.set_xlim([-bw, bw])
            self.ax.set_ylim([-bw, bw])
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.fig.canvas.draw()
            plt.pause(.01)
            i += 1

    def _print_junction(self):
        bw = self.game_params['board_width']
        lw = self.game_params['lane_width']

        # road marks
        plt.vlines(lw, -bw, -lw)
        plt.vlines(-lw, -bw, -lw)
        plt.vlines(-lw, lw, bw)
        plt.vlines(lw, lw, bw)
        plt.hlines(-lw, -bw, -lw)
        plt.hlines(-lw, lw, bw)
        plt.hlines(lw, -bw, -lw)
        plt.hlines(lw, lw, bw)

        # lane marks
        plt.vlines(0, -bw, -lw, linestyles='dashed')
        plt.vlines(0, lw, bw, linestyles='dashed')
        plt.hlines(0, -bw, -lw, linestyles='dashed')
        plt.hlines(0, lw, bw, linestyles='dashed')

        # stopping lines
        plt.hlines(-lw, 0, lw, linestyles='solid')
        plt.hlines(-lw-0.01, 0, lw, linestyles='solid')

        plt.hlines(lw, -lw, 0, linestyles='solid')
        plt.hlines(lw+0.01, -lw, 0, linestyles='solid')

        plt.vlines(-lw, -lw, 0, linestyles='solid')
        plt.vlines(-lw-0.01, -lw, 0, linestyles='solid')

        plt.vlines(lw, 0, lw, linestyles='solid')
        plt.vlines(lw+0.01, 0, lw, linestyles='solid')

        # print rails
        # self.ax.add_patch(Circle(xy=(lw,-lw), radius=lw/2, edgecolor='k', facecolor='none'))
        # self.ax.add_patch(Circle(xy=(-lw,-lw), radius=3*lw/2, edgecolor='k', facecolor='none'))


    def print_info_line(self, iter):

        buf = '%s Training ctrlr 1: iter %d, cost %0.2f, grad_mean %0.4f, grad_max %0.2f, abs norm: %0.2f, lr %0.4f, ' \
              'cost progress %0.4f, cost accel %0.4f, cost accident %0.4f, cost row %0.4f, cost steer %0.4f, cost rail %0.4f\n' % \
                                    (datetime.datetime.now(), iter, self.avg_cost, self.grad_mean, self.max_grad, self.param_abs_norm, self.lr_func(iter),
                                     self.avg_cost_progress, self.avg_cost_accel, self.avg_cost_accident, self.avg_cost_row, self.avg_cost_steer, self.avg_cost_rail)
        sys.stdout.write('\r'+buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") +('%0.6d.sn'%iter), obj = common.get_params(self.model.params))