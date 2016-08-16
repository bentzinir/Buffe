from mgail import MGAIL
import numpy as np
import tensorflow as tf
import time
import common
import sys
import subprocess
import os

class DRIVER(object):

    def __init__(self, environment):

        self.env = environment

        self.algorithm = MGAIL(environment=self.env)

        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        if self.env.trained_model:
            self.saver.restore(self.sess, self.env.run_dir + '/snapshots/' + self.env.trained_model)
        else:
            self.sess.run(self.init_graph)

        self.run_dir = self.env.run_dir
        self.loss = 0 * np.ones(3)
        self.abs_grad = 0 * np.ones(3)
        self.abs_w = 0 * np.ones(3)
        self.time = 0
        self.run_avg = 0.95
        self.discriminator_policy_switch = 0
        self.disc_acc = 0
        # self.run_simulator()
        np.set_printoptions(precision=3)

    # def run_simulator(self):
    #     subprocess.Popen(self.env.run_dir + "./simulator")

    def train_module(self, module, ind, n_steps):

        scan_0 = self.env.drill_state(expert=self.algorithm.er_expert, start_at_zero=1)

        time_vec = [float(t) for t in range(n_steps)]

        if ind == 2:
            # policy
            run_vals = self.sess.run(fetches=[module.minimize,
                                              module.loss,
                                              module.mean_abs_grad,
                                              module.mean_abs_w,
                                              self.algorithm.scan_returns,
                                              ],
                                     feed_dict={
                                        self.algorithm.scan_0: scan_0,
                                        self.algorithm.time_vec: time_vec,
                                        self.algorithm.gamma: self.env.gamma,
                                        self.algorithm.sigma: self.env.sigma,
                                       })

            self.algorithm.push_er(module=self.algorithm.er_agent, trajectory=run_vals[4])

        else:
            states_, actions, _, states, terminals = self.algorithm.er_agent.sample()
            labels = np.expand_dims(np.concatenate([terminals, terminals]), axis=1)  # stub connection

            states_ = np.squeeze(states_, axis=1)
            states = np.squeeze(states, axis=1)

            if ind == 1:  # discriminator
                state_e, action_e, _, _, _ = self.algorithm.er_expert.sample()
                state_e = np.squeeze(state_e, axis=1)

                states_ = np.concatenate([states_, state_e])
                actions = np.concatenate([actions, action_e])

                # labels (policy/expert) : 0/1, and in 1-hot form: policy-[1,0], expert-[0,1]
                labels_p = np.zeros(shape=(self.algorithm.er_agent.batch_size,))
                labels_e = np.ones(shape=(self.algorithm.er_agent.batch_size,))
                labels = np.expand_dims(np.concatenate([labels_p, labels_e]), axis=1)

            run_vals = self.sess.run(fetches=[module.minimize,
                                              module.loss,
                                              module.mean_abs_grad,
                                              module.mean_abs_w,
                                              module.acc
                                              ],
                                     feed_dict={
                                        self.algorithm.states_: states_,
                                        self.algorithm.actions: actions,
                                        self.algorithm.states: states,
                                        self.algorithm.labels: labels,
                                        })

        self.loss[ind] = self.run_avg * self.loss[ind] + (1-self.run_avg) * np.asarray(run_vals[1])
        self.abs_grad[ind] = self.run_avg * self.abs_grad[ind] + (1-self.run_avg) * np.asarray(run_vals[2])
        self.abs_w[ind] = self.run_avg * self.abs_w[ind] + (1-self.run_avg) * np.asarray(run_vals[3])

        if ind == 1:  # discriminator
            self.disc_acc = self.run_avg * run_vals[4] + (1-self.run_avg) * self.disc_acc

    def train_step(self, itr):

        for k_t in range(self.env.K_T):
            # transition
            self.train_module(self.algorithm.transition, ind=0, n_steps=self.env.n_steps_train)

        if itr > self.env.model_identification_time:
            # discriminator (trains only when policy is strong enough)
            if self.discriminator_policy_switch:
                if self.loss[2] < self.env.halt_disc_p_gap_th or (np.random.uniform(low=0, high=1) > self.env.p_train_disc):
                    for k_d in range(self.env.K_D):
                        self.train_module(self.algorithm.discriminator, ind=1, n_steps=self.env.n_steps_train)
            else:
                for k_p in range(self.env.K_P):
                    # policy
                    self.train_module(self.algorithm.policy, ind=2, n_steps=self.env.n_steps_train)

        # print progress
        if itr % 100 == 0:
            buf = self.env.info_line(itr, self.loss, self.disc_acc)
            sys.stdout.write('\r' + buf)

        # switch discriminator-policy
        if itr % self.env.discr_policy_itrvl == 0:
            self.discriminator_policy_switch = not self.discriminator_policy_switch

    def test_step(self):
        n_steps = self.env.n_steps_test
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.env.drill_state(expert=self.algorithm.er_expert, start_at_zero=1)

        run_vals = self.sess.run(fetches=[self.algorithm.scan_returns,
                                          self.algorithm.policy.mean_abs_grad,
                                          self.algorithm.policy.mean_abs_w
                                          ],
                                 feed_dict={self.algorithm.scan_0: scan_0,
                                            self.algorithm.time_vec: time_vec,
                                            self.algorithm.gamma: self.env.gamma,
                                            self.algorithm.sigma: 0,
                                            })

        self.algorithm.push_er(module=self.algorithm.er_agent, trajectory=run_vals[0])
        self.env.test_trajectory = run_vals[0][:, -1, :]

    def print_info_line(self, itr):
        buf = self.env.info_line(itr, self.loss, self.disc_acc, self.abs_grad, self.abs_w)
        sys.stdout.write('\r' + buf)

    def save_model(self, itr):
        common.save_params(fName=self.run_dir + '/snapshots/' + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % itr),
                           saver=self.saver, session=self.sess)
