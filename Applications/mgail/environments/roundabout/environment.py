import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import common
import subprocess
import traceback
import sys
from target import Target


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

        subprocess.Popen(self.run_dir + "./simulator")

        self.pipe_module = tf.load_op_library(self.run_dir + 'pipe.so')

        plt.ion()
        plt.show()

    def reset(self):
        self.t = 0
        # host
        if 1:
            self.x_h = -1.5 * self.r
        else:
            self.x_h = np.random.uniform(low=-1.5 * self.r, high=0)

        # targets
        self._target_objects = []
        self._build_targets()
        targets, host = self.get_state_grid()
        x_t = np.copy(targets[:, 2])
        x_t[x_t > 0] = x_t[x_t > 0] - 2 * np.pi * self.r
        self.ct_ind = np.argmax(x_t)
        tar = targets[self.ct_ind]
        self.state = np.asarray([self.x_h, x_t[self.ct_ind], self.x_h, x_t[self.ct_ind], self.x_h, x_t[self.ct_ind]])

        return self.state

    def _step(self, a):
        a = np.squeeze(a)
        self.t += 1
        # host
        a = np.clip(a, 0, 2.5)
        x_h = self.x_h + a
        self.x_h = np.clip(x_h, -self.r*np.pi, 0.5*self.r*np.pi)

        # target
        state = self.get_state_grid()
        for target in self._target_objects:
            target.move(self.t, state)

        targets, host = self.get_state_grid()
        x_t = targets[self.ct_ind, 2]

        self.state = np.concatenate([[self.x_h, x_t], self.state[:4]])

        self.reward = np.float32(0)
        self.info = np.float32(0)

        if self.x_h > 0. * self.r * np.pi:
            self.done = True
        else:
            self.done = False

        return self.state.astype(np.float32), self.reward.astype(np.float32), self.done

    def step(self, a, mode):
        if mode == 'tensorflow':
            state, reward, done = tf.py_func(self._step, inp=[a], Tout=[tf.float32, tf.float32, tf.bool], name='env_step_func')
            # DEBUG: flatten state. not sure if correctly
            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            state, reward, done = self._step(a)
        info = 0.
        return state, reward, done, info

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def get_state_grid(self):
        num_targets = len(self._target_objects)
        target_state = np.zeros((num_targets,4))
        for i, target in enumerate(self._target_objects):
            target_state[i, 0] = target.get_speed()
            target_state[i, 1] = target.get_relx()
            target_state[i, 2] = target.get_X()
            target_state[i, 3] = target.get_acceleration()

        host_state = [0., 0., self.x_h]
        return target_state, host_state

    def _build_targets(self):
        # =======================================================================
        # Create Targets
        # =======================================================================
        num_targets = self.n_t  # int(self._config.get('settings').get('targets',NUMBER_OF_TARGETS))
        pi = np.pi
        N = num_targets
        R = self.r
        P = np.arange(0, (2 * pi * R - 1e-10), 2 * pi * R / N) + (np.random.rand(N) - 0.5) * (2 * pi * R / N) / 2
        relX = np.hstack((np.diff(P), P[0] + 2 * pi * R - P[N - 1]))
        V = 10 + np.zeros(relX.size);
        X = P + R * pi / 2;
        X[X > pi * R] = X[X > pi * R] - 2 * pi * R

        self.is_aggressive = np.asarray([0] * num_targets)

        for i in xrange(num_targets):
            target = Target(self, i, relX[i], X[i])
            self._target_objects.append(target)
            self.is_aggressive[i] = target._is_aggressive

    def drill_state(self, expert, start_at_zero):

        valid_starts = np.where(expert.terminals[:expert.count] == 1)[0] + 2
        num = np.random.randint(low=0, high=valid_starts.shape[0])
        ind = valid_starts[num]
        state = expert.states[ind - 1]
        return state

        # # drill initial state from expert data
        # scan_0 = np.zeros(shape=(self.scan_batch, self.scan_size), dtype=np.float32)
        # indices = []
        # if start_at_zero is True:
        #     valid_starts = np.where(expert.terminals[:expert.count] == 1)[0] + 2
        # else:
        #     # valid_starts = np.where((expert.terminals[:expert.count] != 1) * (expert.states[:expert.count][:, self.x_field[0]] < 0))[0]
        #     valid_starts = np.where(expert.terminals[:expert.count] != 1)[0]
        #


        # for i in range(self.scan_batch):
        #     num = np.random.randint(low=0, high=valid_starts.shape[0])
        #     ind = valid_starts[num]
        #
        #     state = expert.states[ind-1]
        #     action = expert.actions[ind]
        #
        #     # state
        #     scan_0[i, self.state_field] = state
        #     # action
        #     scan_0[i, self.action_field] = action
        #     # cost
        #     scan_0[i, self.cost_field] = np.zeros(self.cost_size)
        #
        #     # DEBUG: add agg info of closest target
        #     ct_ind = state[-1]
        #     scan_0[i, 29] = state[self.is_aggressive_field[ct_ind]]
        #     indices.append(ind)
        #
        # return scan_0, indices

    def slice_scan(self, scan_vals):
        # only save examples where closest target is approaching (x_ct<=0)
        x_ct = scan_vals[:, self.x_field[1]]
        x_h = scan_vals[:, self.x_field[0]]
        valid_states = (x_ct <= 0) * (x_h <= 0)
        states = scan_vals[valid_states][:, self.state_field]
        actions = scan_vals[valid_states][:, self.action_field.squeeze()]
        return states, actions

    def render(self):

        state = np.squeeze(self.state)

        r = self.r

        x_h = state[0]

        x_h = (x_h > 0) * r * np.array([np.cos(x_h / r - np.pi / 2), np.sin(x_h / r - np.pi / 2)]) +\
              (x_h <= 0) * np.array([0, -r + x_h])

        targets, host = self.get_state_grid()
        x_t = targets[:, 2]

        x_t = [r * np.cos(x_t / r - np.pi / 2), r * np.sin(x_t / r - np.pi / 2)]

        v_t = targets[:, 0]

        aggressive = self.is_aggressive

        self.ax.clear()

        self.ax.set_title(('Sample Trajectory time: %d') % self.t)
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

    def _connect(self, game_params):
        self.dt = game_params['dt']
        self.alphas = np.asarray([1.])
        self.r = game_params['r']
        self.two_pi_r = 2 * np.pi * game_params['r']
        self.num_targets = game_params['num_of_targets']
        self.host_speed = game_params['host_speed']
        self.target_speed = game_params['target_speed']
        self.n_t = game_params['num_of_targets']

        self.state_size = 6
        self.action_size = 1
        self.action_space = np.asarray([None] * self.action_size)

    def _train_params(self):
        self.name = 'roundabout'
        self.best_model = 'backup/2016-08-23-17-04-108000.sn'
        self.trained_model = None
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert-data/expert-2016-11-01-16-54.bin'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.kill_itr = self.n_train_iters
        self.reward_kill_th = -1
        self.test_interval = 2000
        self.n_steps_test = 100
        self.vis_flag = True
        self.model_identification_time = 100000
        self.save_models = True
        self.config_dir = None
        self.tbptt = False
        self.success_th = 3500
        self.n_episodes_expert = 500

        # Main parameters to play with:
        self.reset_itrvl = 10000
        self.n_reset_iters = 5000
        self.collect_experience_interval = 30
        self.n_steps_train = 25
        self.discr_policy_itrvl = 500
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_loss_w = 1e-0
        self.er_agent_size = 100000
        self.total_trans_err_allowed = 1000  # 1e-0
        self.trans_loss_th = 2e-2
        self.max_trans_iters = 2

        self.t_size = [400, 200]
        self.d_size = [100, 50]
        self.p_size = [75, 25]

        self.t_lr = 0.0005
        self.d_lr = 0.001
        self.p_lr = 0.0005

        self.w_std = 0.25

        self.noise_intensity = 3.
        self.do_keep_prob = 0.8

        # Parameters i don't want to play with
        self.disc_as_classifier = True
        self.sae_hidden_size = 80
        self.sae_beta = 3.
        self.sae_rho = 0.01
        self.sae_batch = 25e3
        self.use_sae = False
