import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import common


class ENVIRONMENT(object):

    def __init__(self):

        r = 1.
        game_params = {
            'r': r,
            'dt': 0.15,
            'v_0': 0.2,
            'num_of_targets': 5,
            'dist_in_seconds': 3.,
            'alpha_accel': 0.5,
            'alpha_progress': 0.5,
            'alpha_accident': 0.5,
            'p_aggressive': 0.5,
            'x_goal': 0.5 * r * np.pi,
            'require_distance': 0.25 * r * np.pi,
            'host_length': 0.125 * r * np.pi,
            'gamma': 0.95,
        }

        self._connect(game_params)

        self._train_params()

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

        plt.ion()
        plt.show()

    def record_expert(self, module):
        for i in xrange(self.n_episodes_expert):
            states, actions = self.play_expert()
            rewards = np.zeros(shape=self.batch_size)
            terminals = np.zeros(shape=self.batch_size)
            terminals[-1] = 1
            module.add(actions=actions, rewards=rewards, states=states, terminals=terminals)

            # visualization
            if 0:
                self.ax.clear()
                # loop over trajectory points and plot marks
                for s in states:
                    self.ax.scatter(s[2], s[3], s=40, marker='.', color='k')
                self.ax.set_xlim(xmin=-self.L, xmax=2 * self.L)
                self.ax.set_ylim(ymin=-self.L, ymax=2 * self.L)
                self.fig.canvas.draw()
            print("Processed trajectory %d/%d") % (i, self.n_episodes_expert)

    def _move_controller(self, t, state):
        target_state, host_state = state

        two_pi_r = 2*numpy.pi*R

        x_t_ = target_state[self.id][2] #X
        relx_= target_state[self.id][1] #Relx
        v_t_= target_state[self.id][0] #Speed

        next_id = (self.id + 1)%len(target_state)
        next_target = target_state[next_id]
        next_x_t_ = next_target[2]
        next_v_t = next_target[0]

        relx = next_x_t_ - x_t_

        # fix the jump between -pi and +pi
        relx = (relx>0) *relx + (relx<=0)*(two_pi_r + relx)


        x_h_ = host_state[2]
        relx_to_host = x_h_ - x_t_

        is_host_approaching = (x_h_ > -7) * (x_h_ < 2) * (x_t_ > -two_pi_r/4) * (x_t_ < 0) * ((next_x_t_ > 0) + (next_x_t_ < x_t_) *  (relx > 2))

        is_host_on_route = (x_h_ > 0)

        is_host_cipv = (x_h_ > x_t_) * (relx_to_host < relx)

        accel_default = 5*(relx - 2*v_t_)

        accel_host_on_route = is_host_cipv * 5 * (relx_to_host - 1.5*v_t_) + (1-is_host_cipv) * accel_default

        accel_is_aggressive = max(3, 3*(relx_to_host - 1.5*v_t_))

        accel_not_aggressive = relx_to_host - 1.5*v_t_

        accel_host_approaching = self._is_aggressive * accel_is_aggressive + (1 - self._is_aggressive) * accel_not_aggressive

        accel_host_not_approaching = is_host_on_route * accel_host_on_route + (1 - is_host_on_route) * accel_default

        accel = is_host_approaching * accel_host_approaching + (1- is_host_approaching) * accel_host_not_approaching

        v_t = max(0, v_t_ + DT * accel)
        x_t = x_t_ + DT * v_t

        self._accel = accel
        self._speed = v_t
        self._relx = relx
        self._X = x_t
        if self._X > R*numpy.pi: self._X = self._X - 2*numpy.pi*R

    def play_expert(self, noise=1):

        x_ = np.asarray([[0]])
        v_ = np.asarray([[0]])
        actions = np.asarray([[0]])
        a_min = float(self.L) / 20
        a_max = float(self.L) / 10

        for i in xrange(self.n_steps_train):
            state = _move_controller(self, t, state_)

        states = np.concatenate([v_, x_], axis=1)

        return states, actions

    def step(self, state_t_, a):
        # 0. slice previous state
        v_h_ = tf.slice(state_t_, [self.v_h_field[0]], [1])
        x_h_ = tf.slice(state_t_, [self.x_h_field[0]], [1])
        v_t_ = tf.slice(state_t_, [self.v_t_field[0]], [self.n_t])
        x_t_ = tf.slice(state_t_, [self.x_t_field[0]], [self.n_t])
        a_t_ = tf.slice(state_t_, [self.a_t_field[0]], [self.n_t])
        is_aggressive = tf.slice(state_t_, [self.is_aggressive_field[0]], [self.n_t])

        # 1. simulator logic
        next_x_t_ = tf.concat(concat_dim=0, values=[tf.slice(x_t_, [1], [self.n_t - 1]), tf.slice(x_t_, [0], [1])],
                              name='state')

        relx = next_x_t_ - x_t_

        # fix the jump between -pi and +pi
        relx = tf.mul(tf.to_float(relx >= 0), relx) + tf.mul(tf.to_float(relx < 0), self.two_pi_r + relx)

        relx_to_host = x_h_ - x_t_

        is_host_cipv = tf.mul(tf.mul(tf.to_float((x_h_ > -0.5 * self.host_length)), tf.to_float(x_h_ > x_t_)),
                              tf.to_float(relx_to_host < relx))

        # # If host CIPV - Change relx to him
        relx = tf.mul(is_host_cipv, tf.mul(tf.to_float(x_h_ > 0), tf.to_float(x_h_ - x_t_))) + tf.mul((1 - is_host_cipv),
                                                                                                      relx)

        is_host_approaching = tf.mul(tf.mul(
            tf.mul(tf.mul(tf.to_float(x_h_ > -self.dist_in_seconds * self.host_length), tf.to_float(x_h_ <= -0.5 * self.host_length)),
                   tf.to_float(x_t_ < 0)), tf.to_float(x_t_ > -0.25 * self.two_pi_r)),
                                     tf.to_float(tf.logical_or(next_x_t_ > 0, next_x_t_ < x_t_)))

        accel_default = tf.mul(5., relx) - tf.mul(10., v_t_)

        accel_is_aggressive = tf.maximum(3., tf.mul(3., relx_to_host - tf.mul(self.dist_in_seconds, v_t_)))

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

        # x_h = tf.mul(tf.to_float(x_h >= 0.5 * self.two_pi_r), x_h - self.two_pi_r) +\
        #       tf.mul(tf.to_float(x_h < 0.5 * self.two_pi_r), x_h)

        x_h = tf.clip_by_value(x_h, -self.r, 0.499*self.two_pi_r)

        # 2.2 targets
        v_t = tf.maximum(0., v_t_ + tf.mul(self.dt, accel))
        x_t = x_t_ + tf.mul(self.dt, v_t)
        a_t = v_t - v_t_

        # apply [-pi,pi] discontinuity
        x_t = tf.mul(tf.to_float(x_t >= 0.5 * self.two_pi_r), x_t - self.two_pi_r) + tf.mul(x_t, tf.to_float(
            x_t < 0.5 * self.two_pi_r))

        return tf.concat(concat_dim=0, values=[v_h, x_h, v_t, x_t, a_t], name='state')

    def step_loss(self, state, action, time):
        # cost:
        x_h = tf.slice(state, [0, self.x_h_field[0]], [-1, 1])
        x_t = tf.slice(state, [0, self.x_t_field[0]], [-1, self.n_t])

        # 0. smooth acceleration policy
        cost_accel = tf.square(action)
        cost_accel_d = tf.mul(tf.pow(self.gamma, time), cost_accel)

        # 1. forcing the host to move forward (until the right point of the roundabout)
        cost_prog = tf.square(self.x_goal - x_h)
        cost_prog_d = tf.mul(tf.pow(self.gamma, time), cost_prog)
        cost_prog_d = tf.squeeze(cost_prog_d, squeeze_dims=[1])

        # 2. keeping distance from vehicles ahead
        # distance to other vehicles
        x_abs_diffs = tf.abs(x_h - x_t)

        # punish only vehicles closer than "require distance"
        cost_acci = tf.nn.relu(self.require_distance - x_abs_diffs)

        # punish only w.r.t vehicles ahead
        cost_acci = tf.mul(cost_acci, tf.to_float(x_h < x_t))

        # sum over all vehicles
        cost_acci = tf.reduce_sum(cost_acci)

        # punish only when host is inside the roundabout (or very close to enter)
        cost_acci = tf.mul(cost_acci, tf.to_float(x_h > -0.5 * self.host_length))

        cost_acci_d = tf.mul(tf.pow(self.gamma, time), cost_acci)
        cost_acci_d = tf.squeeze(cost_acci_d, squeeze_dims=[1])

        return tf.transpose(tf.pack(values=[cost_accel_d, cost_prog_d, cost_acci_d], name='scan_return'))

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

    def pack_scan(self, state, action, cost, scan):
        is_aggressive = tf.slice(scan, [self.is_aggressive_field[0]], [self.n_t])
        return tf.concat(concat_dim=0, values=[state, is_aggressive, tf.squeeze(action, squeeze_dims=[0]), cost], name='scan_pack')

    def drill_state(self):
        n_t = self.n_t
        state_0 = np.zeros(self.scan_size, dtype=np.float32)
        state_0[self.v_h_field] = np.zeros(1)
        state_0[self.x_h_field] = np.random.uniform(low=-2*self.r, high=-self.r, size=1)
        state_0[self.v_t_field] = self.v_0 * np.ones(shape=n_t)
        state_0[self.x_t_field] = np.sort(np.random.uniform(low=-self.r*np.pi, high=self.r*np.pi, size=n_t))
        state_0[self.a_t_field] = np.zeros(self.n_t)
        state_0[self.is_aggressive_field] = np.random.binomial(n=1, p=self.p_agg, size=n_t)
        state_0[self.action_field] = np.zeros(1)
        state_0[self.cost_field] = np.zeros(self.cost_size)

        return state_0

    def slice_scan(self, scan_vals):
        N = scan_vals.shape[0]
        actions = scan_vals[:, self.action_field.squeeze()]
        reward = scan_vals[:, self.cost_field.squeeze()]
        state = scan_vals[:, self.state_field]
        return actions, reward, state, N

    def play_trajectory(self):
        r = self.r
        v_h_test = self.test_trajectory[:, self.v_h_field[0]]
        x_h_test = self.test_trajectory[:, self.x_h_field[0]]
        v_t_test = self.test_trajectory[:, self.v_t_field]
        x_t_test = self.test_trajectory[:, self.x_t_field]
        costs = self.test_trajectory[:, self.cost_field]

        is_aggressive = self.test_trajectory[:, self.is_aggressive_field][0]

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
        for i, (x_h, v_h, x_t, v_t, c) in enumerate(zip(x_h_world, v_h_world, x_t_world, v_t_world, costs)):
            self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))
            circle = plt.Circle((0,0), radius = self.r, edgecolor='k', facecolor='none', linestyle='dashed')
            self.fig.gca().add_artist(circle)

            self.ax.scatter(x_h[0],x_h[1], s=180, marker='o', color='r')
            self.ax.annotate(v_h, (x_h[0],x_h[1]))

            for tx, ty, v_, is_ag in zip(x_t[0], x_t[1], v_t, is_aggressive):
                if is_ag:
                    self.ax.scatter(tx, ty, s=180, marker='o', color='b')
                    self.ax.annotate(v_, (tx, ty))
                else:
                    self.ax.scatter(tx, ty, s=180, marker='o', color='g')
                    self.ax.annotate(v_, (tx, ty))

            # print "costs: %s" % c
            self.ax.set_xlim(xmin=-2*self.r, xmax=2*self.r)
            self.ax.set_ylim(ymin=-3*self.r, ymax=2*self.r)
            self.fig.canvas.draw()
            time.sleep(0.01)

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alpha_accel = game_params['alpha_accel']
        self.alpha_progress = game_params['alpha_progress']
        self.alpha_accident = game_params['alpha_accident']
        #self.alphas = np.asarray([self.alpha_accel, self.alpha_progress, self.alpha_accident])
        self.alphas = np.asarray([self.alpha_accel])
        self.two_pi_r = 2 * np.pi * game_params['r']
        self.num_targets = game_params['num_of_targets']
        self.x_goal = game_params['x_goal']
        self.require_distance = game_params['require_distance']
        self.dist_in_seconds = game_params['dist_in_seconds']
        self.v_0 = game_params['v_0']
        self.p_agg = game_params['p_aggressive']
        self.r = game_params['r']
        self.host_length = game_params['host_length']
        self.n_t = game_params['num_of_targets']
        self.gamma = game_params['gamma']

        self.f_ptr = 0
        common.add_field(self, 'v_h', 1)
        common.add_field(self, 'x_h', 1)
        common.add_field(self, 'v_t', self.n_t)
        common.add_field(self, 'x_t', self.n_t)
        common.add_field(self, 'a_t', self.n_t)
        common.add_field(self, 'is_aggressive', self.n_t)
        common.add_field(self, 'action', 1)
        common.add_field(self, 'cost', 1)

        self.state_field = np.arange(self.v_h_field[0], self.a_t_field[-1]+1)
        self.state_size = self.a_t_field[-1]+1
        self.scan_size = self.f_ptr
        self.cost_size = self.cost_field.shape[0]
        self.state_size = 2 + 3*self.n_t
        self.action_size = 1

    def _train_params(self):
        self.name = 'Linemove_2d'
        self.trained_model = None
        self.expert_data = None
        self.n_train_iters = 1000000
        self.test_interval = 2000
        self.n_steps_train = 100
        self.n_steps_test = 100
        self.n_episodes_expert = 500
        self.model_identification_time = 0
        self.discr_policy_itrvl = 200
        self.K = 1
        self.gamma = 0.99
        self.batch_size = 32
