import numpy as np
from svg_1 import SVG_1
import tensorflow as tf
import datetime
import time
import common
import sys

class DRIVER(object):

    def __init__(self, simulator, train_params, trained_model, sn_dir):

        self.train_params = train_params

        self.simulator = simulator

        # with tf.device('/cpu:0')

        self.algorithm = SVG_1(self.simulator)
        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        if trained_model:
            self.saver.restore(self.sess,trained_model)
        else:
            self.sess.run(self.init_graph)

        self.sn_dir = sn_dir
        self.loss = 999 * np.ones(3)
        self.abs_grad = 999 * np.ones(3)
        self.abs_w = 999 * np.ones(3)
        np.set_printoptions(precision=3)

    def collect_experience(self, n_steps):

        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.simulator.drill_state()
        n_std = self.algorithm.ER.n_std

        returned_vals = self.sess.run(fetches=[self.algorithm.scan_returns],
                                        feed_dict={self.algorithm.scan_0: scan_0,
                                                   self.algorithm.time_vec: time_vec,
                                                   self.algorithm.noise_std: n_std,
                                                   })
        return scan_0, returned_vals[0]

    def train_step(self, iter):
        n_steps = self.train_params['n_steps_train']
        _, returned_vals = self.collect_experience(n_steps)
        self.algorithm.push_ER(returned_scan_vals=returned_vals)
        states_, actions, rewards, states, terminals, p_k = self.algorithm.ER.sample()
        n_std = self.algorithm.ER.n_std

        returned_train_vals = self.sess.run(fetches=[self.algorithm.minimize_t,
                                                     self.algorithm.minimize_c,
                                                     self.algorithm.minimize_p,
                                                     self.algorithm.loss_t,
                                                     self.algorithm.loss_c,
                                                     self.algorithm.loss_p,
                                                     self.algorithm.transition.mean_abs_grad,
                                                     self.algorithm.critic.mean_abs_grad,
                                                     self.algorithm.policy.mean_abs_grad,
                                                     self.algorithm.transition.mean_abs_w,
                                                     self.algorithm.critic.mean_abs_w,
                                                     self.algorithm.policy.mean_abs_w,
                                                     ],
                                            feed_dict={
                                                       self.algorithm.state_: np.squeeze(states_),
                                                       self.algorithm.action: actions,
                                                       self.algorithm.state_e: np.squeeze(states),
                                                       self.algorithm.p_k: p_k,
                                                       self.algorithm.noise_std: n_std,
                                                       })

        self.loss = 0.95 * self.loss + 0.05 * np.asarray(returned_train_vals[3:6])
        self.abs_grad = 0.95 * self.abs_grad + 0.05 * np.asarray(returned_train_vals[6:9])
        self.abs_w = 0.95 * self.abs_w + 0.05 * np.asarray(returned_train_vals[9:12])

        # update target weights
        if iter % self.algorithm.critic.solver_params['target_update_freq'] == 0:
            self.sess.run(self.algorithm.critic.copy_target_weights_op)

        # print progress
        if iter % 10 == 0:
            buf = "processing iter: %d, loss(transition,critic,policy): %s" % (iter, self.loss)
            sys.stdout.write('\r' + buf)

    def test_step(self):
        n_steps = self.train_params['n_steps_test']

        self.simulator.scan_0, self.simulator.returned_test_vals = self.collect_experience(n_steps)

    def print_info_line(self, iter):
        buf = '%s Training ctrlr 1: iter %d, loss: %s, grads: %s, weights: %s\n' % \
                                    (datetime.datetime.now(), iter, self.loss, self.abs_grad, self.abs_w)
        sys.stdout.write('\r' + buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % iter),
                           saver=self.saver, session=self.sess)
