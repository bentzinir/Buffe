import tensorflow as tf
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import common
import subprocess


class ENVIRONMENT(object):

    def __init__(self, run_dir):

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

        self.run_dir = run_dir

        common.compile_modules(self.run_dir)

        subprocess.Popen(self.run_dir + "./simulator")

        self.pipe_module = tf.load_op_library(self.run_dir + 'pipe.so')

        plt.ion()
        plt.show()

    # def compile_modules(self):
    #     cwd = os.getcwd()
    #     os.chdir(self.run_dir)
    #     os.system('g++ -std=c++11 simulator.c -o simulator')
    #     os.system('g++ -std=c++11 -shared pipe.cc -o pipe.so -fPIC -I $TF_INC')
    #     os.chdir(cwd)

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

        for i in xrange(self.n_steps_test):
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

    def _step(self, state_, action):

        # 0. slice previous state
        v_ = tf.slice(state_, [self.v_field[0]], [2])
        x_ = tf.slice(state_, [self.x_field[0]], [2])

        v = v_ + tf.mul(self.dt, tf.squeeze(action, squeeze_dims=[0]))

        # clip host speed to the section -v0,v0]
        v = tf.clip_by_value(v, -self.v_max, self.v_max)

        x = x_ + self.dt * v

        # x = tf.clip_by_value(x, 0, self.L)

        return tf.concat(concat_dim=0, values=[v, x], name='state')

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

    def pack_scan(self, state, action, cost, scan=None):
        return tf.concat(concat_dim=1, values=[state, action, cost], name='scan_pack')

    def pack_state_for_simulator(self, scan_, action):
        state_ = tf.slice(scan_, [0, 0], [-1, self.state_size])
        return tf.concat(concat_dim=1, values=[state_, action], name='state_action')

    def drill_state(self, expert, start_at_zero=0):
        state_0 = np.zeros(shape=(self.scan_batch, self.scan_size), dtype=np.float32)
        if start_at_zero:
            valid_starts = np.where(expert.terminals == 1)[0] + 1
        else:
            valid_starts = np.where((expert.terminals != 1))[0]

        for i in range(self.scan_batch):
            ind = np.random.randint(low=0, high=valid_starts.shape[0])
            state = expert.states[valid_starts[ind]]
            state_0[i, self.v_field] = state[:2]
            state_0[i, self.x_field] = state[2:4]
            state_0[i, self.cost_field] = np.zeros(self.cost_size)

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

        scat_agent = self.ax.scatter(x_test[0][0], x_test[0][1], s=180, marker='o', color='r')
        scat_expert = self.ax.scatter(x_expert[0][0], x_expert[0][1], s=180, marker='o', color='b')

        # loop over trajectory points and plot
        for i, (x, v, x_e) in enumerate(zip(x_test, v_test, x_expert)):
            # self.ax.clear()
            self.ax.set_title(('Sample Trajectory\n time: %d' % i))
            scat_agent.set_offsets(x)
            scat_expert.set_offsets(x_e)

            self.ax.set_xlim(xmin=-self.L, xmax=2 * self.L)
            self.ax.set_ylim(ymin=-self.L, ymax=2 * self.L)
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
        self.name = 'linemove_2D'
        self.trained_model = None
        self.expert_data = '2016-08-11-14-00.bin'
        self.n_train_iters = 1000000
        self.test_interval = 1000
        self.n_steps_test = 100
        self.n_episodes_expert = 500
        self.action_size = 2
        self.model_identification_time = 0

        # Main parameters to play with:
        self.n_steps_train = 30 #  65-5
        self.discr_policy_itrvl = 200
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 32  # (discriminator / transition)
        self.scan_batch = 1  #  (policy) MUST BE CHANGED IN SIMULATOR AS WELL !!!!
        self.sigma = 0.01
        self.max_p_gap = 100
        self.er_agent_size = 30000
        self.halt_disc_p_gap_th = 5
        self.p_train_disc = 0.9

