from mgail import MGAIL
import numpy as np
import tensorflow as tf
import time
import common
import sys


class DRIVER(object):

    def __init__(self, environment, configuration, trained_model, sn_dir, expert_data):

        self.env = environment

        self.config = configuration

        self.algorithm = MGAIL(environment=self.env,
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

        self.sn_dir = sn_dir
        self.loss = 0 * np.ones(3)
        self.abs_grad = 0 * np.ones(3)
        self.abs_w = 0 * np.ones(3)
        self.time = 0
        self.discriminator_policy_switch = 1

        np.set_printoptions(precision=3)

    def train_module(self, module, ind, n_steps):

        scan_0 = self.simulator.drill_state()

        time_vec = [float(t) for t in range(n_steps)]

        state_, action, _, state, terminals = self.algorithm.er_agent.sample()

        if ind == 2:
            # policy
            train_vals = self.sess.run(fetches=[module.minimize,
                                                module.loss,
                                                module.mean_abs_grad,
                                                module.mean_abs_w,
                                                self.algorithm.scan_returns,
                                                ],
                                       feed_dict={
                                           self.algorithm.scan_0: scan_0,
                                           self.algorithm.time_vec: time_vec,
                                       })
        else:
            # transition / discriminator
            state_e, action_e, (_) = self.algorithm.er_expert.sample()

            train_vals = self.sess.run(fetches=[module.minimize,
                                                module.loss,
                                                module.mean_abs_grad,
                                                module.mean_abs_w,
                                                ],
                                       feed_dict={
                                            self.algorithm.state_: np.squeeze(state_),
                                            self.algorithm.action: action,
                                            self.algorithm.state: np.squeeze(state),
                                            self.algorithm.state_e: state_e,
                                            self.algorithm.action_e: np.squeeze(action_e),
                                            self.algorithm.gamma: self.gamma,
                                             })

        self.loss[ind] = self.run_avg * self.loss[ind] + (1-self.run_avg) * np.asarray(train_vals[1])
        self.abs_grad[ind] = self.run_avg * self.abs_grad[ind] + (1-self.run_avg) * np.asarray(train_vals[2])
        self.abs_w[ind] = self.run_avg * self.abs_w[ind] + (1-self.run_avg) * np.asarray(train_vals[3])

        self.algorithm.push_ER(returned_scan_vals=train_vals)

    def train_step(self, itr):

        for k in range(self.config.K):
            # transition
            self.train_module(self.algorithm.transition, ind=0)

            if itr > self.config.model_identification_time:
                if self.discriminator_policy_switch:
                    # discriminator
                    self.train_module(self.algorithm.discriminator, ind=1)
                else:
                    # policy
                    self.train_module(self.algorithm.policy, ind=2)

        # print progress
        if itr % 100 == 0:
            buf = self.config.train_buf_line(itr, self.loss, self.time, self.gamma)
            sys.stdout.write('\r' + buf)

        # switch discriminator-policy
        if itr % self.config.discriminator_policy_itrvl == 0:
            self.discriminator_policy_switch = not self.discriminator_policy_switch

    def test_step(self):
        n_steps = self.config.n_steps_test
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.simulator.drill_state()
        returned_test_vals = self.sess.run(fetches=[self.algorithm.scan_returns,
                                                    self.algorithm.policy.mean_abs_grad,
                                                    self.algorithm.policy.mean_abs_w
                                                    ],
                                           feed_dict={self.algorithm.scan_0: scan_0,
                                                      self.algorithm.time_vec: time_vec})

        self.env.scan_0 = scan_0
        self.env.returned_test_vals = returned_test_vals

    def print_info_line(self, itr):
        buf = self.config.test_buf_line(itr, self.loss, self.abs_grad, self.abs_w, self.gamma)
        sys.stdout.write('\r' + buf)

    def save_model(self, itr):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % itr),
                           saver=self.saver, session=self.sess)
