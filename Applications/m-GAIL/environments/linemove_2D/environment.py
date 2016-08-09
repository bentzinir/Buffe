import tensorflow as tf
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import common


class ENVIRONMENT(object):

    def __init__(self):

        game_params = {
            'L': 2,
            'dt': 0.15,
            'v_0': 0.,
            'v_max': 0.5,
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
            rewards = np.zeros_like(actions[:, 0])
            terminals = np.zeros_like(actions[:, 0])
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

    def play_expert(self, noise=1):

        x_goals = np.asarray([[self.L, 0],
                             [self.L, self.L],
                             [0, self.L],
                             [0, 0]])
        goal = 0

        x_ = np.asarray([[0, 0]])
        v_ = np.asarray([[0, 0]])
        actions = np.asarray([[0, 0]])
        a_min = float(self.L)/20
        a_max = float(self.L)/10

        for i in xrange(self.n_steps_train):
            d = x_goals[goal] - x_[-1]
            d_abs_clip = np.clip(abs(d), 0, a_max)
            a = np.sign(d) * d_abs_clip

            # add noise
            if noise == 1:
                a += np.random.normal(scale=a_min/20)

            # transition
            v = v_[-1] + self.dt * a
            v = np.clip(v, -self.v_max, self.v_max)
            x = x_[-1] + self.dt * v

            x_ = np.append(x_, np.expand_dims(x, 0), axis=0)
            v_ = np.append(v_, np.expand_dims(v, 0), axis=0)
            actions = np.append(actions, np.expand_dims(a, 0), axis=0)

            if np.linalg.norm(d) < float(self.L)/10:
                goal += 1
                goal = goal % 4

        states = np.concatenate([v_, x_], axis=1)

        return states, actions

    def step(self, state_, action):

        # 0. slice previous state
        v_ = tf.slice(state_, [self.v_field[0]], [2])
        x_ = tf.slice(state_, [self.x_field[0]], [2])

        v = v_ + tf.mul(self.dt, tf.squeeze(action, squeeze_dims=[0]))

        # clip host speed to the section -v0,v0]
        v = tf.clip_by_value(v, -self.v_max, self.v_max)

        x = x_ + self.dt * v

        # x = tf.clip_by_value(x, 0, self.L)

        return tf.concat(concat_dim=0, values=[v, x], name='state')

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
        return tf.concat(concat_dim=0, values=[state, tf.squeeze(action, squeeze_dims=[0]), cost], name='scan_pack')

    def drill_state(self):
        state_0 = np.zeros(self.scan_size, dtype=np.float32)
        state_0[self.v_field] = 0
        state_0[self.x_field] = 0
        state_0[self.cost_field] = np.zeros(self.cost_size)

        return state_0

    def slice_scan(self, scan_vals):
        N = scan_vals.shape[0]
        actions = scan_vals[:, self.action_field.squeeze()]
        reward = scan_vals[:, self.cost_field.squeeze()]
        state = scan_vals[:, self.state_field]
        return actions, reward, state, N

    def reference_contour(self):
        states, actions = self.play_expert(noise=0)

        self.ax.clear()

        # loop over trajectory points and plot marks
        for s in states:
            self.ax.scatter(s[2], s[3], s=40, marker='.', color='k')
        self.ax.set_xlim(xmin=-self.L, xmax=2 * self.L)
        self.ax.set_ylim(ymin=-self.L, ymax=2 * self.L)
        self.fig.canvas.draw()
        return states

    def play_trajectory(self):
        # plot reference contour
        x_expert = self.reference_contour()[:, 2:]

        v_test = self.test_trajectory[:, self.v_field]
        x_test = self.test_trajectory[:, self.x_field]

        self.ax.set_title('Sample Trajectory')

        # x_world = []
        # v_world = []
        # for x_h, v_h in zip(x_test, v_test):
        #     x, y = np.asarray([x_h, 0])
        #     x_world.append([x, y])
        #     v_world.append(v_h)

        scat_agent = self.ax.scatter(x_test[0][0], x_test[0][1], s=180, marker='o', color='r')
        scat_expert = self.ax.scatter(x_expert[0][0], x_expert[0][1], s=180, marker='o', color='b')

        # loop over trajectory points and plot
        for i, (x, v, x_e) in enumerate(zip(x_test, v_test, x_expert)):
            # self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))
            scat_agent.set_offsets(x)
            scat_expert.set_offsets(x_e)

            # self.ax.annotate(np.linalg.norm(v), (x[0], x[1]))

            self.ax.set_xlim(xmin=-self.L, xmax=2 * self.L)
            self.ax.set_ylim(ymin=-self.L, ymax=2 * self.L)
            self.fig.canvas.draw()
            time.sleep(0.01)

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alpha_accel = 1.
        self.alphas = np.asarray([self.alpha_accel])
        self.v_0 = game_params['v_0']
        self.v_max = game_params['v_max']
        self.L = game_params['L']

        self.f_ptr = 0
        common.add_field(self, 'v', 2)
        common.add_field(self, 'x', 2)
        common.add_field(self, 'action', 2)
        common.add_field(self, 'cost', 1)

        self.state_field = np.arange(self.v_field[0], self.x_field[-1]+1)
        self.scan_size = self.f_ptr
        self.cost_size = self.cost_field.shape[0]
        self.state_size = 4
        self.action_size = 2

    def _train_params(self):
        self.name = 'Linemove_2d'
        self.trained_model = None
        self.expert_data = '2016-08-09-08-25.bin'
        self.n_train_iters = 1000000
        self.test_interval = 2000
        self.n_steps_train = 100
        self.n_steps_test = 100
        self.n_episodes_expert = 500
        self.action_size = 2
        self.model_identification_time = 0
        self.discr_policy_itrvl = 200
        self.K = 1
        self.gamma = 0.99
        self.batch_size = 32
        self.alpha_accel = 0.5,
        self.alpha_progress = 0.5,
        self.alpha_accident = 0.5,
        self.w_kl = 1e-4
        self.sigma = 0.01

    def info_line(self, itr, loss, discriminator_acc, abs_grad=None, abs_w=None):
        if abs_grad is not None:
            buf = '%s Training: iter %d, loss: %s, disc_acc: %f, grads: %s, weights: %s\n' % \
                  (datetime.datetime.now(), itr, loss, discriminator_acc, abs_grad, abs_w)
        else:
            buf = "processing iter: %d, loss(transition,discriminator,policy): %s, disc_acc: %f" % (
                itr, loss, discriminator_acc)
        return buf
