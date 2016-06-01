from controller import CONTROLLER
import matplotlib.pyplot as plt
import tensorflow as tf
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

        self.game_params = game_params
        self.arch_params = arch_params
        self.solver_params = solver_params

        self.learning_rate_func = common.create_lr_func(solver_params)

        # with tf.device('/cpu:0'):
        # self.time_vec = tf.placeholder("float",shape=arch_params['n_steps_train'])
        self.time_vec = tf.placeholder("float")
        self.lr = tf.placeholder("float")
        self.n_t = self.game_params['num_of_targets']
        self.game_params['state_dim'] = (1 + # v_h_0
                                         1 + # x_h_0
                                         2 + # costs
                                           + self.n_t * (
                                             1 + # v_t_0
                                             1 + # x_t_0
                                             1 + # a_t_0
                                             1 # is aggresive
                                            )
                                         )

        self.state_0 = tf.placeholder("float", shape=self.game_params['state_dim'])

        self.sn_dir = sn_dir

        self.model = CONTROLLER(self.state_0, self.time_vec, self.lr, self.game_params, self.arch_params, self.solver_params, params)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(self.init)

        self.avg_cost = 999
        self.grad_mean = 0
        self.param_abs_norm = 0

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=2)

        plt.ion()
        plt.show()

    def _drill_state(self):
        r = self.game_params['r']
        n_t = self.game_params['num_of_targets']
        v_h_0 = np.zeros(1) # np.asarray([self.game_params['v_0']])
        x_h_0 = - 1 * np.asarray([r])
        v_t_0 = self.game_params['v_0'] * np.ones(shape=n_t)
        x_t_0 = np.sort(np.random.uniform(low= -r*np.pi, high = r*np.pi, size=n_t))
        a_t_0 = np.zeros_like(v_t_0)
        costs = np.zeros(2)
        is_aggressive = np.random.binomial(n=1, p=self.game_params['p_aggressive'], size=n_t)
        state_0 = np.concatenate([v_h_0,
                                  x_h_0,
                                  v_t_0,
                                  x_t_0,
                                  a_t_0,
                                  is_aggressive,
                                  costs]).astype(np.float32)
        return state_0

    def train_step(self, iter):
        n_steps = self.arch_params['n_steps_train']
        time_vec = [float(t) for t in range(n_steps)]
        state_0 = self._drill_state()
        lr = self.learning_rate_func(iter,self.solver_params)

        returned_train_vals = self.sess.run(fetches=[self.model.apply_grads,
                                                     self.model.cost],
                                            feed_dict={self.state_0:state_0,
                                                       self.time_vec:time_vec,
                                                       self.lr:lr})

        self.avg_cost = 0.95 * self.avg_cost + 0.05 * returned_train_vals[1]

        if iter % 10 == 0:
            buf = "processing iter: %d, cost: (%0.2f)" % (iter, self.avg_cost)
            sys.stdout.write('\r' + buf)

    def test_step(self):
        n_steps = self.arch_params['n_steps_test']
        time_vec = [float(t) for t in range(n_steps)]
        state_0 = self._drill_state()
        returned_test_vals = self.sess.run(fetches=[self.model.v_h,
                                                    self.model.x_h,
                                                    self.model.v_t,
                                                    self.model.x_t,
                                                    self.model.mean_abs_grad,
                                                    self.model.mean_abs_w
                                                    ],
                                           feed_dict={self.state_0:state_0,
                                                    self.time_vec: time_vec})

        self.v_h_test = returned_test_vals[0].squeeze()
        self.x_h_test = returned_test_vals[1].squeeze()
        self.v_t_test = returned_test_vals[2].squeeze()
        self.x_t_test = returned_test_vals[3].squeeze()
        # self.grad_mean = returned_test_vals[4]
        # self.param_abs_norm = returned_test_vals[5]

        self.is_aggressive_test = state_0[2+3*self.n_t : 2+4*self.n_t]

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
            x,y = (x_h > 0)*r*np.array([np.cos(x_h/r-np.pi/2),np.sin(x_h/r-np.pi/2)]) + (x_h <= 0)*np.array([0,-r+x_h])
            x_h_world.append([x,y])
            v_h_world.append(v_h)

        x_t_world = []
        v_t_world = []
        for x_t, v_t in zip(x_t_test, v_t_test):
            x = r*np.cos(x_t/r - np.pi/2)
            y = r*np.sin(x_t/r - np.pi/2)
            x_t_world.append([x,y])
            v_t_world.append(v_t)

        # loop over trajectory points and plot
        for i, (x_h, v_h, x_t, v_t) in enumerate(zip(x_h_world, v_h_world, x_t_world, v_t_world)):
            self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))
            circle = plt.Circle((0,0), radius = self.game_params['r'], edgecolor='k', facecolor='none', linestyle='dashed')
            self.fig.gca().add_artist(circle)

            self.ax.scatter(x_h[0],x_h[1], s=180, marker='o', color='r')
            self.ax.annotate(v_h, (x_h[0],x_h[1]))

            for tx,ty, v_, is_ag in zip(x_t[0], x_t[1], v_t, is_aggressive):
                if is_ag:
                    self.ax.scatter(tx,ty, s=180, marker='o', color='b')
                    self.ax.annotate(v_, (tx,ty))
                else:
                    self.ax.scatter(tx,ty, s=180, marker='o', color='g')
                    self.ax.annotate(v_, (tx,ty))

            self.ax.set_xlim(xmin=-2*self.game_params['r'], xmax=2*self.game_params['r'])
            self.ax.set_ylim(ymin=-3*self.game_params['r'], ymax=2*self.game_params['r'])
            self.fig.canvas.draw()
            time.sleep(0.01)

    def print_info_line(self, iter):

        buf = '%s Training ctrlr 1: iter %d, cost: %0.4f, grad_mean %0.12f, abs norm %0.2f, lr %0.6f\n' % \
                                    (datetime.datetime.now(), iter, self.avg_cost, self.grad_mean, self.param_abs_norm, self.learning_rate_func(iter, self.solver_params))
        sys.stdout.write('\r' + buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % iter), saver=self.saver, session=self.sess)
