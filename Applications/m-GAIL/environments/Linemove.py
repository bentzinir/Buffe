import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

class ENVIRONMENT(object):

    def __init__(self):

        game_params = {
            'L': 5,
            'dt': 0.15,
            'v_0': 0.,
            'v_max': 0.5,
        }

        self._connect(game_params)

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

        plt.ion()
        plt.show()

    def step(self, state_t_, a):
        # 0. slice previous state
        v_ = tf.slice(state_t_, [self.v_field[0]], [1])
        x_ = tf.slice(state_t_, [self.x_field[0]], [1])

        v = v_ + tf.mul(self.dt, a)

        # clip host speed to the section -v0,v0]
        v = tf.clip_by_value(v, -self.v_max, self.v_max)

        x = x_ + self.dt * v

        x = tf.clip_by_value(x, 0, self.L)

        return tf.concat(concat_dim=0, values=[v, x], name='state')

    def expert(self, state):
        x = tf.slice(state, [self.x_field[0]], [1])
        a = tf.mul(tf.to_float(x < self.L/2), 0.1) + tf.mul(tf.to_float(x >= self.L/2), -0.1)
        return tf.expand_dims(a,1)

    def weight_costs(self, costs):
        return tf.reduce_sum(tf.mul(costs, self.alphas[:self.cost_size]))

    def total_loss(self, scan_returns):
        # slice different cost measures
        costs = tf.slice(scan_returns, [0, self.cost_field[0]], [-1, self.cost_size])

        # average over time
        cost_vec = tf.reduce_mean(costs, reduction_indices=0)

        # for printings
        cost = tf.reduce_sum(cost_vec)

        # for training
        cost_weighted = self.weight_costs(cost_vec)

        return cost, cost_weighted

    def pack_scan(self, state, action, cost):
        return tf.concat(concat_dim=0, values=[state, action, cost], name='scan_pack')

    def drill_state(self):
        state_0 = np.zeros(self.scan_size, dtype=np.float32)
        state_0[self.v_field] = np.zeros(1)
        state_0[self.x_field] = np.zeros(1)
        state_0[self.cost_field] = np.zeros(self.cost_size)

        return state_0

    def slice_scan(self, scan_vals):
        N = scan_vals.shape[0]
        actions = scan_vals[:, self.action_field.squeeze()]
        reward = scan_vals[:, self.cost_field.squeeze()]
        state = scan_vals[:, self.state_field]
        return actions, reward, state, N

    def play_trajectory(self):
        v_test = self.test_trajectory[:, self.v_field[0]]
        x_test = self.test_trajectory[:, self.x_field[0]]
        costs = self.test_trajectory[:, self.cost_field]

        self.ax.set_title('Sample Trajectory')

        x_world = []
        v_world = []
        for x_h, v_h in zip(x_test, v_test):
            x,y = np.asarray([x_h, 0])
            x_world.append([x,y])
            v_world.append(v_h)

        # loop over trajectory points and plot
        for i, (x, v, c) in enumerate(zip(x_world, v_world, costs)):
            self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))

            self.ax.scatter(x[0], x[1], s=180, marker='o', color='r')
            self.ax.annotate(v, (x[0], x[1]))

            self.ax.set_xlim(xmin=-self.L, xmax=self.L)
            self.ax.set_ylim(ymin=-self.L, ymax=self.L)
            self.fig.canvas.draw()
            #print 'cost: %f' % c
            time.sleep(0.01)

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alpha_accel = 1.
        self.alphas = np.asarray([self.alpha_accel])
        self.v_0 = game_params['v_0']
        self.v_max = game_params['v_max']
        self.L = game_params['L']

        self.f_ptr = 0
        self.add_field('v', 1)
        self.add_field('x', 1)
        self.add_field('action',1)
        self.add_field('cost', 1)

        self.state_field = np.arange(self.v_field[0], self.x_field[-1]+1)
        self.state_size = 2
        self.scan_size = self.f_ptr
        self.cost_size = self.cost_field.shape[0]
        self.action_size = 1

    def add_field(self, name, size):
        setattr(self, name+'_field', np.arange(self.f_ptr, self.f_ptr+size))
        self.f_ptr += size
