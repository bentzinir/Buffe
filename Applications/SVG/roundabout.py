import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class ROUNDABOUT(object):

    def __init__(self):

        game_params = {
            'r': 1.,
            'dt': 0.1,
            'v_0': 0.9,
            'num_of_targets': 5,
            'dist_in_seconds': 1.8,
            'alpha_accel': 0.1,
            'alpha_progress': 1.,
            'alpha_accident': 0.,
            'require_distance': 0.1,
            'p_aggressive': 0.5,
            'host_length': 0.1,
            'alpha_cost_a': 0.5,
            'alpha_cost_t': 0.99,
            'gamma': 0.9,
        }

        self._connect(game_params)

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

        plt.ion()
        plt.show()

    def step(self, state_t_, a):
        # 0. slice previous state
        v_h_ = tf.slice(state_t_, [0], [1])
        x_h_ = tf.slice(state_t_, [1], [1])
        v_t_ = tf.slice(state_t_, [2], [self.n_t])
        x_t_ = tf.slice(state_t_, [2 + 1 * self.n_t], [self.n_t])
        a_t_ = tf.slice(state_t_, [2 + 2 * self.n_t], [self.n_t])
        is_aggressive = tf.slice(state_t_, [2 + 3 * self.n_t], [self.n_t])

        # 1. simulator logic
        next_x_t_ = tf.concat(concat_dim=0, values=[tf.slice(x_t_, [1], [self.n_t - 1]), tf.slice(x_t_, [0], [1])],
                              name='state')

        relx = next_x_t_ - x_t_

        # x_t_ = tf.Print(x_t_, [x_t_], message='x_t_:', summarize=self.n_t)

        # fix the jump between -pi and +pi
        relx = tf.mul(tf.to_float(relx >= 0), relx) + tf.mul(tf.to_float(relx < 0), self.two_pi_r + relx)

        relx_to_host = x_h_ - x_t_

        is_host_cipv = tf.mul(tf.mul(tf.to_float((x_h_ > -0.5 * self.host_length)), tf.to_float(x_h_ > x_t_)),
                              tf.to_float(relx_to_host < relx))

        # # If host CIPV - Change relx to him
        relx = tf.mul(is_host_cipv, tf.mul(tf.to_float(x_h_ > 0), tf.to_float(x_h_ - x_t_))) + tf.mul((1 - is_host_cipv),
                                                                                                      relx)

        is_host_approaching = tf.mul(tf.mul(
            tf.mul(tf.mul(tf.to_float(x_h_ > -1.5 * self.host_length), tf.to_float(x_h_ <= -0.5 * self.host_length)),
                   tf.to_float(x_t_ < 0)), tf.to_float(x_t_ > -0.25 * self.two_pi_r)),
                                     tf.to_float(tf.logical_or(next_x_t_ > 0, next_x_t_ < x_t_)))

        accel_default = tf.mul(5., relx) - tf.mul(10., v_t_)

        accel_is_aggressive = tf.maximum(3., tf.mul(3., relx_to_host - tf.mul(1.5, v_t_)))

        accel_not_aggressive = tf.div(0.5 - v_t_, self.dt)

        accel_host_approaching = tf.mul(is_aggressive, accel_is_aggressive) + tf.mul(1 - is_aggressive,
                                                                                     accel_not_aggressive)

        accel = tf.mul(is_host_approaching, accel_host_approaching) + tf.mul(1 - is_host_approaching, accel_default)

        # 2. exact next state
        # 2.1 host
        v_h = v_h_ + tf.mul(self.dt, a)

        # clip host speed to the section [0,v0]
        v_h = tf.clip_by_value(v_h, 0, self.v_0)

        x_h = x_h_ + self.dt * v_h

        x_h = tf.mul(tf.to_float(x_h >= 0.5 * self.two_pi_r), x_h - self.two_pi_r) + tf.mul(
            tf.to_float(x_h < 0.5 * self.two_pi_r), x_h)

        # 2.2 targets
        v_t_e = tf.maximum(0., v_t_ + tf.mul(self.dt, accel))
        x_t_e = x_t_ + tf.mul(self.dt, v_t_e)
        a_t_e = v_t_e - v_t_

        # 3. learn the transition model between states
        v_t_a = v_t_e
        x_t_a = x_t_e
        a_t_a = a_t_e

        # 4. prediction noise
        n_v_t = v_t_e - v_t_a
        n_x_t = x_t_e - x_t_a
        n_a_t = a_t_e - a_t_a

        # 5. disconnect the gradient of the noise signals
        tf.stop_gradient(n_v_t)
        tf.stop_gradient(n_x_t)
        tf.stop_gradient(n_a_t)

        # 6. add the noise to the approximation
        v_t = v_t_a + n_v_t
        x_t = x_t_a + n_x_t
        a_t = a_t_a + n_a_t

        # apply [-pi,pi] discontinuity
        x_t = tf.mul(tf.to_float(x_t >= 0.5 * self.two_pi_r), x_t - self.two_pi_r) + tf.mul(x_t, tf.to_float(
            x_t < 0.5 * self.two_pi_r))

        state_t = tf.concat(concat_dim=0, values=[v_h, x_h, v_t, x_t, a_t, is_aggressive], name='state')

        return state_t

    def step_cost(self, state, action, time):
        # cost:
        x_h = tf.slice(state, [1], [1])
        x_t = tf.slice(state, [2 + self.n_t], [self.n_t])

        # 0. smooth acceleration policy
        cost_accel = tf.square(action)
        cost_accel_d = tf.mul(tf.pow(self.gamma, time), cost_accel)
        cost_accel_d = tf.expand_dims(cost_accel_d, 0)

        # 1. forcing the host to move forward (until the right point of the roundabout)
        cost_prog = tf.square(0.25 * self.two_pi_r - x_h)
        cost_prog_d = tf.mul(tf.pow(self.gamma, time), cost_prog)

        # 2. keeping distance from vehicles ahead
        # distance to other vehicles
        x_abs_diffs = tf.abs(x_h - x_t)

        # punish only vehicles closer than "require distance"
        cost_acci = tf.nn.relu(self.require_distance - x_abs_diffs)

        # punish only w.r.t ahead vehicles
        # cost_acci = tf.mul(cost_acci, tf.to_float(x_h < x_t))

        # sum over all vehicles
        cost_acci = tf.reduce_sum(cost_acci)

        # punish only when host is inside the roundabout (or very close to enter)
        cost_acci = tf.mul(cost_acci, tf.to_float(x_h > -0.5 * self.host_length))

        cost_acci_d = tf.mul(tf.pow(self.gamma, time), cost_acci)

        return tf.concat(concat_dim=0, values=[cost_accel_d, cost_prog_d, cost_acci_d], name='scan_return')

    def total_cost(self, scan_returns):
        # slice different cost measures
        costs_accel = tf.slice(scan_returns, [0, 2 + 4 * self.n_t + 0], [-1, 1])
        costs_prog = tf.slice(scan_returns,  [0, 2 + 4 * self.n_t + 1], [-1, 1])
        costs_acci = tf.slice(scan_returns,  [0, 2 + 4 * self.n_t + 2], [-1, 1])

        # costs_acci = tf.Print(costs_acci,[costs_acci],message='cost_acci: ',summarize=50)

        # average over time
        cost_accel = tf.reduce_mean(costs_accel)
        cost_prog = tf.reduce_mean(costs_prog)
        cost_acci = tf.reduce_mean(costs_acci)

        # for printings
        cost = cost_accel + cost_prog + cost_acci

        # for training
        cost_weighted = tf.mul(cost_accel, self.alpha_accel) + tf.mul(cost_prog, self.alpha_progress) + tf.mul(cost_acci, self.alpha_accident)

        return cost, cost_weighted

    def drill_state(self):
        n_t = self.n_t
        v_h_0 = np.zeros(1)
        x_h_0 = - 1 * np.asarray([self.r])
        v_t_0 = self.v_0 * np.ones(shape=n_t)
        x_t_0 = np.sort(np.random.uniform(low= -self.r*np.pi, high = self.r*np.pi, size=n_t))
        a_t_0 = np.zeros_like(v_t_0)
        costs = np.zeros(3)
        is_aggressive = np.random.binomial(n=1, p=self.p_agg, size=n_t)
        state_0 = np.concatenate([v_h_0,
                                  x_h_0,
                                  v_t_0,
                                  x_t_0,
                                  a_t_0,
                                  is_aggressive,
                                  costs]).astype(np.float32)
        return state_0

    def play_trajectory(self):

        r = self.r
        v_h_test = self.returned_test_vals[:, 0]
        x_h_test = self.returned_test_vals[:, 1]
        v_t_test = self.returned_test_vals[:, 2:2 + self.n_t]
        x_t_test = self.returned_test_vals[:, 2 + self.n_t:2 + 2 * self.n_t]

        is_aggressive = self.state_0[2+3*self.n_t : 2+4*self.n_t]

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
            circle = plt.Circle((0,0), radius = self.r, edgecolor='k', facecolor='none', linestyle='dashed')
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

            self.ax.set_xlim(xmin=-2*self.r, xmax=2*self.r)
            self.ax.set_ylim(ymin=-3*self.r, ymax=2*self.r)
            self.fig.canvas.draw()
            time.sleep(0.01)

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alpha_accel = game_params['alpha_accel']
        self.alpha_progress = game_params['alpha_progress']
        self.alpha_accident = game_params['alpha_accident']
        self.two_pi_r = 2 * np.pi * game_params['r']
        self.num_targets = game_params['num_of_targets']
        self.require_distance = game_params['require_distance']
        self.v_0 = game_params['v_0']
        self.p_agg = game_params['p_aggressive']
        self.r = game_params['r']
        self.host_length = game_params['host_length']
        self.n_t = game_params['num_of_targets']
        self.gamma = game_params['gamma']
        self.state_dim = (1 +  # v_h_0
                            1 +  # x_h_0
                            3 +  # costs
                            + self.n_t * (
                                1 +  # v_t_0
                                1 +  # x_t_0
                                1 +  # a_t_0
                                1  # is aggresive
                                )
                          )
