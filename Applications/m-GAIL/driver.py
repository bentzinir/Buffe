from mgail import MGAIL
import numpy as np
import tensorflow as tf
import time
import common
import sys


class DRIVER(object):

    def __init__(self, environment, configuration, trained_model, run_dir, expert_data):

        self.env = environment

        self.config = configuration

        self.algorithm = MGAIL(environment=self.env,
                               configuration=self.config,
                               state_size=self.env.state_size,
                               action_size=self.env.action_size,
                               scan_size=self.env.scan_size,
                               expert_data=expert_data
                               )

        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        if trained_model:
            self.saver.restore(self.sess, trained_model)
        else:
            self.sess.run(self.init_graph)

        self.run_dir = run_dir
        self.loss = 0 * np.ones(3)
        self.abs_grad = 0 * np.ones(3)
        self.abs_w = 0 * np.ones(3)
        self.time = 0
        self.run_avg = 0.95
        self.discriminator_policy_switch = 0
        self.disc_acc = 0

        np.set_printoptions(precision=3)

    def train_module(self, module, ind, n_steps):

        scan_0 = self.env.drill_state()

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
                                        self.algorithm.gamma: self.config.gamma,
                                       })

            self.algorithm.push_er(module=self.algorithm.er_agent, trajectory=run_vals[4])

        else:
            # transition / discriminator
            state_, action, _, state, terminals = self.algorithm.er_agent.sample()
            state_e, action_e, _, _, _ = self.algorithm.er_expert.sample()

            # labels (policy/expert) : 0/1, and in vector form: policy-[1,0], expert-[0,1]
            labels_p = np.zeros_like(action)
            labels_e = np.ones_like(action_e)

            run_vals = self.sess.run(fetches=[module.minimize,
                                              module.loss,
                                              module.mean_abs_grad,
                                              module.mean_abs_w,
                                              module.acc
                                              ],
                                     feed_dict={
                                        self.algorithm.state_: np.squeeze(state_),
                                        self.algorithm.action: action,
                                        self.algorithm.state: np.squeeze(state),

                                        self.algorithm.labels: np.expand_dims(np.concatenate([labels_p, labels_e]), axis=1),
                                        self.algorithm.states: np.squeeze(np.concatenate([state_, state_e])),
                                        self.algorithm.actions: np.concatenate([action, action_e]),
                                        })

        self.loss[ind] = self.run_avg * self.loss[ind] + (1-self.run_avg) * np.asarray(run_vals[1])
        self.abs_grad[ind] = self.run_avg * self.abs_grad[ind] + (1-self.run_avg) * np.asarray(run_vals[2])
        self.abs_w[ind] = self.run_avg * self.abs_w[ind] + (1-self.run_avg) * np.asarray(run_vals[3])

        if ind == 1:  # discriminator
            self.disc_acc = self.run_avg * run_vals[4] + (1-self.run_avg) * self.disc_acc

    def train_step(self, itr):

        for k in range(self.config.K):
            # transition
            self.train_module(self.algorithm.transition, ind=0, n_steps=self.config.n_steps_train)

            if itr > self.config.model_identification_time:
                if self.discriminator_policy_switch:
                    # discriminator
                    self.train_module(self.algorithm.discriminator, ind=1, n_steps=self.config.n_steps_train)
                else:
                    # policy
                    self.train_module(self.algorithm.policy, ind=2, n_steps=self.config.n_steps_train)

        # print progress
        if itr % 100 == 0:
            buf = self.config.info_line(itr, self.loss, self.disc_acc)
            sys.stdout.write('\r' + buf)

        # switch discriminator-policy
        if itr % self.config.discr_policy_itrvl == 0:
            self.discriminator_policy_switch = not self.discriminator_policy_switch

    def test_step(self):
        n_steps = self.config.n_steps_test
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.env.drill_state()

        run_vals = self.sess.run(fetches=[self.algorithm.scan_returns,
                                          self.algorithm.policy.mean_abs_grad,
                                          self.algorithm.policy.mean_abs_w
                                          ],
                                 feed_dict={self.algorithm.scan_0: scan_0,
                                            self.algorithm.time_vec: time_vec,
                                            self.algorithm.gamma: self.config.gamma,
                                            })

        self.env.test_trajectory = run_vals[0]
        self.algorithm.push_er(module=self.algorithm.er_agent, trajectory=self.env.test_trajectory)

    def record_expert(self):
        n_steps = self.config.n_steps_train
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.env.drill_state()

        run_vals = self.sess.run(fetches=[self.algorithm.scan_returns,
                                          self.algorithm.policy.mean_abs_grad,
                                          self.algorithm.policy.mean_abs_w
                                          ],
                                 feed_dict={self.algorithm.scan_0: scan_0,
                                            self.algorithm.time_vec: time_vec,
                                            self.algorithm.gamma: self.config.gamma,
                                            })

        self.env.test_trajectory = run_vals[0]
        self.algorithm.push_er(module=self.algorithm.er_expert, trajectory=self.env.test_trajectory)

    def print_info_line(self, itr):
        buf = self.config.info_line(itr, self.loss, self.disc_acc, self.abs_grad, self.abs_w)
        sys.stdout.write('\r' + buf)

    def save_model(self, itr):
        common.save_params(fName=self.run_dir + '/snapshots/' + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % itr),
                           saver=self.saver, session=self.sess)
