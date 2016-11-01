import tensorflow as tf
import time
from ER import ER
import matplotlib.pyplot as plt
import numpy as np
import common
import sys
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

        self.run_dir = run_dir

        if self.collect_data:
            self.record_expert()
            sys.exit(0)

        self._init_display()

        # subprocess.Popen(self.run_dir + "./simulator")

        # self.pipe_module = tf.load_op_library(self.run_dir + 'pipe.so')

        plt.ion()
        plt.show()

    def record_expert(self):
        er = ER(memory_size=self.er_agent_size,
                state_dim=self.state_size,
                action_dim=self.action_size,
                reward_dim=1,  # stub connection
                batch_size=self.batch_size,
                history_length=1)

        all_states = np.zeros((1, self.state_size))
        all_actions = np.zeros((1, self.action_size))

        for i in xrange(self.n_episodes_expert):
            # DEBUG !!! remove noise
            states, actions = self.play_expert(noise=1)
            rewards = np.zeros_like(actions[:, 0])
            terminals = np.zeros_like(actions[:, 0])
            terminals[-1] = 1
            er.add(actions=actions, rewards=rewards, next_states=states, terminals=terminals)

            all_states = np.append(all_states, states, axis=0)
            all_actions = np.append(all_actions, actions, axis=0)

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

        er.states_mean = np.mean(all_states, axis=0)
        er.actions_mean = np.mean(all_actions, axis=0)

        er.states_std = np.std(all_states, axis=0)
        er.actions_std = np.std(all_actions, axis=0)

        common.save_er(directory=self.run_dir, module=er)

    def play_expert(self, noise=1):
        x_goals = np.asarray([[self.L, 0],
                             [self.L, self.L],
                             [0, self.L],
                             [0, 0]]).astype(np.float32)
        goal = 0

        x_ = np.asarray([[0, 0]]).astype(np.float32)
        v_ = np.asarray([[0, 0]]).astype(np.float32)
        actions = np.asarray([[0, 0]]).astype(np.float32)
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

    def _step(self, a):
        a = np.squeeze(a)
        self.t += 1
        v = self.state[:2] + self.dt * a
        v = np.clip(v, -self.v_max, self.v_max)
        x = self.state[2:] + self.dt * v
        self.state = np.concatenate([v, x])
        if self.t >= 100:
            self.done = True
        else:
            self.done = False
        reward = np.float32(0.1)
        return self.state.astype(np.float32), reward, self.done

    def step(self, a, mode):
        if mode == 'tensorflow':
            state, reward, done = tf.py_func(self._step, inp=[a], Tout=[tf.float32, tf.float32, tf.bool], name='env_step_func')
            # DEBUG: flatten state. not sure if correctly
            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            state, reward, done = self._step(a)
        info = 0
        return state, reward, done, info

    def reset(self):
        self.t = 0
        x_ = [0, 0]
        v_ = [0, 0]
        self.state = np.concatenate([v_, x_]).astype(np.float32)
        return self.state

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

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

    def render(self):
        self.ax.set_title(('Sample Trajectory\n time: %d' % self.t))
        x_test = self.state[2:]
        v_test = self.state[:2]
        self.scat_agent.set_offsets(x_test)
        self.scat_expert.set_offsets(self.x_expert[self.t])
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
        self.state_size = 4
        self.action_size = 2
        self.action_space = np.asarray([None]*self.action_size)

    def _init_display(self):
        self.reset()
        self.t = 0
        if self.vis_flag:
            self.fig = plt.figure()
            self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=2)
            self.x_expert = self.reference_contour()[:, 2:]
            self.scat_agent = self.ax.scatter(self.state[2], self.state[3], s=180, marker='o', color='r')
            self.scat_expert = self.ax.scatter(self.x_expert[0][0], self.x_expert[0][1], s=180, marker='o', color='b')

    def _train_params(self):
        self.name = 'linemove_2D'
        self.collect_data = False
        self.trained_model = None
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert_data/expert-2016-10-26-11-43.bin'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.kill_itr = self.n_train_iters
        self.reward_kill_th = -1
        self.test_interval = 2000
        self.n_steps_test = 100
        self.vis_flag = True
        self.model_identification_time = 0
        self.save_models = True
        self.config_dir = None
        self.tbptt = False
        self.success_th = 3500
        self.n_episodes_expert = 500

        # Main parameters to play with:
        self.reset_itrvl = 10000
        self.n_reset_iters = 5000
        self.collect_experience_interval = 15
        self.n_steps_train = 25
        self.discr_policy_itrvl = 500
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_loss_w = 1e-0
        self.er_agent_size = 50000
        self.total_trans_err_allowed = 1000# 1e-0
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
        self.dropout_ratio = 0.4

        # Parameters i don't want to play with
        self.disc_as_classifier = True
        self.sae_hidden_size = 80
        self.sae_beta = 3.
        self.sae_rho = 0.01
        self.sae_batch = 25e3
        self.use_sae = False
