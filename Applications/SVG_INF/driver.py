from svg_inf import SVG_INF
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

        # Main graph: includes the algorithm and policy
        self.algorithm = SVG_INF(self.simulator)
        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        if trained_model:
            self.saver.restore(self.sess,trained_model)
        else:
            self.sess.run(self.init_graph)

        # Second graph: includes transition model
        self.graph_t = tf.Graph()
        with self.graph_t.as_default():
            pass

        self.sn_dir = sn_dir
        self.policy_loss = 999
        self.transition_loss = 999
        self.grad_mean = 0
        self.param_abs_norm = 0

    def train_step(self, iter):
        n_steps = self.train_params['n_steps_train']
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.simulator.drill_state()
        states_, actions, rewards, states, terminals = self.algorithm.ER.sample()

        returned_train_vals = self.sess.run(fetches=[self.algorithm.minimize_controller,
                                                     self.algorithm.minimize_transition,
                                                     self.algorithm.scan_returns,
                                                     self.algorithm.policy_loss,
                                                     self.algorithm.transition_loss],
                                            feed_dict={self.algorithm.scan_0: scan_0,
                                                       self.algorithm.time_vec: time_vec,
                                                       self.algorithm.state_: states_,
                                                       self.algorithm.action: actions,
                                                       self.algorithm.state_e: states,
                                                       })

        self.algorithm.push_ER(returned_scan_vals=returned_train_vals[2])

        self.policy_loss = 0.95 * self.policy_loss + 0.05 * returned_train_vals[3]

        self.transition_loss = 0.95 * self.transition_loss + 0.05 * returned_train_vals[4]

        if iter % 10 == 0:
            buf = "processing iter: %d, loss(policy,transition): (%0.6f,%0.6f)" % (iter, self.policy_loss, self.transition_loss)
            sys.stdout.write('\r' + buf)

    def test_step(self):
        n_steps = self.train_params['n_steps_test']
        time_vec = [float(t) for t in range(n_steps)]
        scan_0 = self.simulator.drill_state()
        returned_test_vals = self.sess.run(fetches=[self.algorithm.scan_returns,
                                                    self.algorithm.policy.mean_abs_grad,
                                                    self.algorithm.policy.mean_abs_w
                                                    ],
                                           feed_dict={self.algorithm.scan_0:scan_0,
                                                    self.algorithm.time_vec: time_vec})

        self.simulator.returned_test_vals = returned_test_vals[0]
        self.grad_mean = returned_test_vals[1]
        self.param_abs_norm = returned_test_vals[2]

        self.simulator.scan_0 = scan_0

    def print_info_line(self, iter):
        buf = '%s Training ctrlr 1: iter %d, policy loss: %0.3f, transition_loss: %0.3f, grad_mean %0.2f, abs norm %0.2f\n' % \
                                    (datetime.datetime.now(), iter, self.policy_loss, self.transition_loss, self.grad_mean, self.param_abs_norm)
        sys.stdout.write('\r' + buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % iter),
                           saver=self.saver, session=self.sess)
