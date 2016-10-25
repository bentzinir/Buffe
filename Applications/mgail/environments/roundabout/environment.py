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
        }

        self._connect(game_params)

        self._train_params()

        self.fig = plt.figure()
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)

        self.run_dir = run_dir

        # common.compile_modules(self.run_dir)

        subprocess.Popen(self.run_dir + "./simulator")

        self.pipe_module = tf.load_op_library(self.run_dir + 'pipe.so')

        np.set_printoptions(precision=3)
        np.set_printoptions(linewidth=160)

        plt.ion()
        plt.show()

    def play_expert(self, noise=1):
        pass

    def step(self, scan_, a):
        if 1:
            state_ = tf.slice(scan_, [0, 0], [-1, self.state_size])

            state_action = tf.concat(concat_dim=1, values=[state_, a], name='state_action')

            state_action = tf.reshape(state_action, [-1])

            state_e = self.pipe_module.pipe(state_action)

            state_e = tf.slice(state_e, [0], [self.scan_batch * self.state_size])

            state_e = tf.reshape(state_e, [self.scan_batch, -1])

        else:
            state_ = tf.slice(scan_, [0, 0], [-1, self.env.state_size])

            state_e = self._step(state_=state_, action=a)

        return state_e

    def total_loss(self, scan_returns):

        # slice different cost measures
        costs = tf.slice(scan_returns, [0, 0, self.cost_field[0]], [-1, -1, self.cost_size])

        # penalize only when x_ct is still approaching or x_h is approaching
        x_ct = tf.slice(scan_returns, [0, 0, self.x_field[1]], [-1, -1, 1])
        x_h = tf.slice(scan_returns, [0, 0, self.x_field[0]], [-1, -1, 1])

        valid_ct = tf.less_equal(x_ct, 0)
        costs = tf.mul(tf.cast(valid_ct, "float"), costs)

        valid_h = tf.less_equal(x_h, 0)
        costs = tf.mul(tf.cast(valid_h, "float"), costs)

        # average over time
        cost_vec = tf.reduce_mean(costs, reduction_indices=0)

        # for printings
        cost = tf.reduce_mean(cost_vec)

        # for training
        cost_weighted = cost

        return cost, cost_weighted

    def drill_state(self, expert, start_at_zero):

        # drill initial state from expert data
        scan_0 = np.zeros(shape=(self.scan_batch, self.scan_size), dtype=np.float32)
        indices = []
        if start_at_zero is True:
            valid_starts = np.where(expert.terminals[:expert.count] == 1)[0] + 2
        else:
            # valid_starts = np.where((expert.terminals[:expert.count] != 1) * (expert.states[:expert.count][:, self.x_field[0]] < 0))[0]
            valid_starts = np.where(expert.terminals[:expert.count] != 1)[0]

        for i in range(self.scan_batch):
            num = np.random.randint(low=0, high=valid_starts.shape[0])
            ind = valid_starts[num]

            state = expert.states[ind-1]
            action = expert.actions[ind]

            # state
            scan_0[i, self.state_field] = state
            # action
            scan_0[i, self.action_field] = action
            # cost
            scan_0[i, self.cost_field] = np.zeros(self.cost_size)

            # DEBUG: add agg info of closest target
            ct_ind = state[-1]
            scan_0[i, 29] = state[self.is_aggressive_field[ct_ind]]
            indices.append(ind)

        return scan_0, indices

    def slice_scan(self, scan_vals):
        # only save examples where closest target is approaching (x_ct<=0)
        x_ct = scan_vals[:, self.x_field[1]]
        x_h = scan_vals[:, self.x_field[0]]
        valid_states = (x_ct <= 0) * (x_h <= 0)
        states = scan_vals[valid_states][:, self.state_field]
        actions = scan_vals[valid_states][:, self.action_field.squeeze()]
        return states, actions

    def render(self, state, i, term, disc):

        state = np.squeeze(state)

        r = self.r

        x_h = state[self.x_field[0]]

        x_h = (x_h > 0) * r * np.array([np.cos(x_h / r - np.pi / 2), np.sin(x_h / r - np.pi / 2)]) +\
              (x_h <= 0) * np.array([0, -r + x_h])

        x_t = state[self.x_t_field]
        x_t = [r * np.cos(x_t / r - np.pi / 2), r * np.sin(x_t / r - np.pi / 2)]

        v_t = state[self.v_t_field]

        aggressive = state[self.is_aggressive_field]

        self.ax.clear()

        self.ax.set_title(('Sample Trajectory time: %d, terminal: %d \n p_agent: %f, p_expert: %f, p_gap: %f' %
                           (i, term, disc[0], disc[1], disc[0]-disc[1])))
        circle = plt.Circle((0, 0), radius=self.r, edgecolor='k', facecolor='none', linestyle='dashed')
        self.fig.gca().add_artist(circle)

        self.ax.scatter(x_h[0], x_h[1], s=180, marker='o', color='r')
        # self.ax.annotate(v_h, (x_h[0], x_h[1]))

        for tx, ty, v_, is_ag in zip(x_t[0], x_t[1], v_t, aggressive):
            if is_ag:
                self.ax.scatter(tx, ty, s=180, marker='o', color='b')
                self.ax.annotate(v_, (tx, ty))
            else:
                self.ax.scatter(tx, ty, s=180, marker='o', color='g')
                self.ax.annotate(v_, (tx, ty))

        # print "costs: %s" % c
        self.ax.set_xlim(xmin=-2 * self.r, xmax=2 * self.r)
        self.ax.set_ylim(ymin=-3 * self.r, ymax=2 * self.r)
        self.fig.canvas.draw()
        time.sleep(0.01)

    def play_trajectory(self, module, n_steps, play_expert=False, start_at_zero=True):

        expert = module.algorithm.er_expert

        # drill state
        scan, indices = self.drill_state(expert, start_at_zero=start_at_zero)

        state_ = scan[-1, self.state_field]
        action = scan[-1, self.action_field]

        ind = indices[0]

        term = 0

        disc = np.zeros(2)

        state_buffer = [state_]*3

        self.is_ct_agg = scan[-1, 29]

        for i in range(n_steps):
            # render
            self.render(state_, i, term, disc)

            # act and move
            if play_expert is True:
                ind += 1
                state = expert.states[ind-1]
                action = expert.actions[ind]
                term = expert.terminals[ind]
                print action
            else:
                # DEBUG: add is_aggressive_info to state
                run_vals = module.sess.run(fetches=[module.algorithm.fast_action,
                                                    module.algorithm.fast_state,
                                                    module.algorithm.fast_disc],
                                           feed_dict={module.algorithm.state_: np.tile(state_, (10, 1)),
                                                      module.algorithm.is_ct_agg: np.tile(self.is_ct_agg, (10, 1))})
                action = run_vals[0][-1]
                state = run_vals[1][-1]
                disc = run_vals[2][-1]

                term = 0
                state_buffer.append(state)
                state = self.update_closest_target(state_buffer)

            state_ = state

    def update_closest_target(self, state_buffer):
        state = state_buffer[-1]
        x_ct = state[self.x_field[1]]
        ct_ind = state[self.ct_ind_field][0]

        if x_ct > 0:
            ct_ind = (ct_ind - 1 % self.n_t)
            state[self.ct_ind_field[0]] = ct_ind
            state[1] = state_buffer[-1][self.x_t_field[int(ct_ind)]]
            state[3] = state_buffer[-2][self.x_t_field[int(ct_ind)]]
            state[5] = state_buffer[-3][self.x_t_field[int(ct_ind)]]

            self.is_ct_agg = state[self.is_aggressive_field[ct_ind]]

        return state

    def info_line(self, itr, loss, discriminator_acc, abs_grad=None, abs_w=None, state_action_grads=None, er_count=None):
        if abs_grad is not None:
            buf = '%s Training: iter %d, loss: %s, disc_acc: %f, grads: %s, weights: %s, D_grads: %s, er_count: %d\n' % \
                  (time.strftime("%H:%M:%S"), itr, loss, discriminator_acc, abs_grad, abs_w, state_action_grads, er_count)
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
        self.n_t = game_params['num_of_targets']
        self.history_length = 3

        self.f_ptr = 0
        # state
        common.add_field(self, 'x', 2)
        common.add_field(self, 'x_', 2)
        common.add_field(self, 'x__', 2)
        common.add_field(self, 'v_t', self.n_t)
        common.add_field(self, 'x_t', self.n_t)
        common.add_field(self, 'a_t', self.n_t)
        common.add_field(self, 'is_aggressive', self.n_t)
        common.add_field(self, 'ct_ind', 1)
        # action
        common.add_field(self, 'action', 1)
        # cost (loss)
        common.add_field(self, 'cost', 1)

        self.state_field = np.arange(self.x_field[0], self.ct_ind_field[-1]+1)
        self.state_size = self.state_field.shape[0]
        self.obs_size = 2 * self.history_length
        self.action_size = self.action_field.shape[0]
        self.cost_size = self.cost_field.shape[0]
        self.scan_size = self.f_ptr
        self.trans_predict_size = 2
        # DEBUG: add is_agg info of closest target
        self.scan_size += 1

    def _train_params(self):
        self.name = 'roundabout'
        self.best_model = 'backup/2016-08-23-17-04-108000.sn'
        self.trained_model = None
        self.expert_data = 'expert-2016-09-06-09-37.bin'
        self.n_train_iters = 1000000
        self.test_interval = 2000
        self.n_steps_test = 60
        self.n_episodes_expert = 3000
        self.model_identification_time = 100000

        # Main parameters to play with:
        self.n_steps_train = 5
        self.discr_policy_itrvl = 200
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.9
        self.batch_size = 10  # (discriminator / transition)
        # self.scan_batch = 1  # (policy) MUST BE CHANGED IN SIMULATOR AS WELL !!!! obsolete - we can't work with a batch in real environments
        self.sigma = 0.1  # DEBUG: reduced sigma significantly
        self.max_p_gap = 20
        self.er_agent_size = 500000
        self.halt_disc_p_gap_th = 5
        self.p_train_disc = 0.99
