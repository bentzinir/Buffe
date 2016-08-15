import tensorflow as tf
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import common
import subprocess

class ENVIRONMENT(object):

    def __init__(self, run_dir):

        r = 10.
        game_params = {
            'r': r,
            'dt': 1./9,
            'host_speed': 10/3.6,
            'target_speed': 5.,
            'num_of_targets': 5,
            'p_aggressive': 0.7
        }

        self._connect(game_params)

        self._train_params()

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

        self.run_dir = run_dir

        common.compile_modules(self.run_dir)

        subprocess.Popen(self.run_dir + "./simulator")

        self.pipe_module = tf.load_op_library(self.run_dir + 'pipe.so')

        plt.ion()
        plt.show()

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

    def step(self, scan_, a):
        if 1:
            state_action = self.pack_state_for_simulator(scan_, a)

            state_action = tf.reshape(state_action, [-1])

            state_e = self.pipe_module.pipe(state_action)

            state_e = tf.slice(state_e, [0], [self.scan_batch * self.state_size])

            state_e = tf.reshape(state_e, [self.scan_batch, -1])

        else:
            state_ = tf.slice(scan_, [0, 0], [-1, self.env.state_size])

            state_e = self._step(state_=state_, action=a)

        return state_e

    def _step(self, state_t_, a):
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
        costs = tf.slice(scan_returns, [0, 0, self.cost_field[0]], [-1, -1, self.cost_size])

        # average over time
        cost_vec = tf.reduce_mean(costs, reduction_indices=0)

        # for printings
        cost = tf.reduce_mean(cost_vec)

        # for training
        # cost_weighted = self.weight_costs(cost_vec)
        # DEBUG
        cost_weighted = cost

        return cost, cost_weighted

    def pack_scan(self, state, action, cost, scan):
        is_aggressive = tf.slice(scan, [0, self.is_aggressive_field[0]], [-1, self.n_t])
        return tf.concat(concat_dim=1, values=[state, is_aggressive, action, cost], name='scan_pack')

    def pack_state_for_simulator(self, scan, action):
        gross_state = tf.slice(scan, [0,0], [-1, self.state_size + self.n_t])
        return tf.concat(concat_dim=1, values=[gross_state, action], name='gross_state')

    def drill_state(self, expert, start_at_zero=0):

        # drill initial state from expert data
        state_0 = np.zeros(shape=(self.scan_batch, self.scan_size), dtype=np.float32)
        # DEBUG
        if start_at_zero:
            valid_starts = np.where(expert.terminals == 1)[0] + 1
        else:
            valid_starts = np.where((expert.terminals != 1) * (expert.states[:, 1] <= 0))[0]

        for i in range(self.scan_batch):
            ind = np.random.randint(low=0, high=valid_starts.shape[0])
            state = expert.states[valid_starts[ind]]
            state_0[i, self.v_h_field] = state[0]
            state_0[i, self.x_h_field] = state[1]
            state_0[i, self.v_t_field] = state[2:7]
            state_0[i, self.x_t_field] = state[7:12]
            state_0[i, self.a_t_field] = state[12:17]
            state_0[i, self.is_aggressive_field] = expert.rewards[ind]
            state_0[i, self.action_field] = np.zeros(1)
            state_0[i, self.cost_field] = np.zeros(self.cost_size)

        return state_0

    def slice_scan(self, scan_vals):
        state = scan_vals[:, self.state_field]
        valid_states = state[:, self.x_h_field[0]] < (0.23 * self.two_pi_r)

        actions = scan_vals[valid_states][:, self.action_field.squeeze()]
        reward = scan_vals[valid_states][:, self.cost_field.squeeze()]
        state = scan_vals[valid_states][:, self.state_field]
        N = actions.shape[0]
        return actions, reward, state, N

    def load_trajectory(self, expert):

        n_steps = 1500

        if expert is not None:
            offset = 0 # np.random.randint(low=0, high=expert.count - n_steps - 1)
            v_h = expert.states[offset: offset + n_steps, 0]
            x_h = expert.states[offset: offset + n_steps, 1]
            v_t = expert.states[offset: offset + n_steps, 2:7]
            x_t = expert.states[offset: offset + n_steps, 7:12]
            is_aggressive = expert.rewards[offset: offset + n_steps, :]
        else:
            v_h = self.test_trajectory[:, self.v_h_field[0]]
            x_h = self.test_trajectory[:, self.x_h_field[0]]
            v_t = self.test_trajectory[:, self.v_t_field]
            x_t = self.test_trajectory[:, self.x_t_field]
            is_aggressive = self.test_trajectory[:, self.is_aggressive_field]

        return v_h, x_h, v_t, x_t, is_aggressive

    def play_trajectory(self, expert=None):
        r = self.r

        V_h, X_h, V_t, X_t, Is_aggressive = self.load_trajectory(expert)

        self.ax.set_title('Sample Trajectory')

        x_h_world = []
        v_h_world = []
        for x_h, v_h in zip(X_h, V_h):
            x,y = (x_h > 0)*r*np.array([np.cos(x_h/r-np.pi/2),np.sin(x_h/r-np.pi/2)]) + (x_h <= 0)*np.array([0,-r+x_h])
            x_h_world.append([x,y])
            v_h_world.append(v_h)

        x_t_world = []
        v_t_world = []
        for x_t, v_t in zip(X_t, V_t):
            x = r*np.cos(x_t/r - np.pi/2)
            y = r*np.sin(x_t/r - np.pi/2)
            x_t_world.append([x,y])
            v_t_world.append(v_t)

        # loop over trajectory points and plot
        for i, (x_h, v_h, x_t, v_t, aggressive) in enumerate(zip(x_h_world, v_h_world, x_t_world, v_t_world, Is_aggressive)):
            self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))
            circle = plt.Circle((0,0), radius = self.r, edgecolor='k', facecolor='none', linestyle='dashed')
            self.fig.gca().add_artist(circle)

            self.ax.scatter(x_h[0],x_h[1], s=180, marker='o', color='r')
            self.ax.annotate(v_h, (x_h[0],x_h[1]))

            for tx, ty, v_, is_ag in zip(x_t[0], x_t[1], v_t, aggressive):
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

    def info_line(self, itr, loss, discriminator_acc, abs_grad=None, abs_w=None):
        if abs_grad is not None:
            buf = '%s Training: iter %d, loss: %s, disc_acc: %f, grads: %s, weights: %s\n' % \
                  (datetime.datetime.now(), itr, loss, discriminator_acc, abs_grad, abs_w)
        else:
            buf = "processing iter: %d, loss(transition,discriminator,policy): %s, disc_acc: %f" % (
                itr, loss, discriminator_acc)
        return buf

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alphas = np.asarray([1.])
        self.r = game_params['r']
        self.two_pi_r = 2 * np.pi * game_params['r']
        self.num_targets = game_params['num_of_targets']
        self.host_speed = game_params['host_speed']
        self.target_speed = game_params['target_speed']
        self.p_agg = game_params['p_aggressive']
        self.n_t = game_params['num_of_targets']

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
        self.name = 'roundabout'
        self.best_model = 'backup/2016-08-15-10-34-010000.sn'
        self.trained_model = None
        self.expert_data = '2016-08-14-12-19.bin'
        self.n_train_iters = 1000000
        self.test_interval = 1000
        self.n_steps_test = 100
        self.n_episodes_expert = 3000
        self.model_identification_time = 0

        # Main parameters to play with:
        self.n_steps_train = 100 #  65-5
        self.discr_policy_itrvl = 200
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 40  # (discriminator / transition)
        self.scan_batch = 1  #  (policy) MUST BE CHANGED IN SIMULATOR AS WELL !!!!
        self.sigma = 0.5
        self.max_p_gap = 100
        self.er_agent_size = 30000
        self.halt_disc_p_gap_th = 5
        self.p_train_disc = 0.9
