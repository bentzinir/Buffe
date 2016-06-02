from controller import CONTROLLER
import tensorflow as tf
import datetime
import time
import common
import sys

class CTRL_OPTMZR(object):

    def __init__(self, simulator, arch_params, solver_params, trained_model, sn_dir):

        params=None

        if trained_model:
            params = common.load_params(trained_model)

        self.simulator = simulator
        self.arch_params = arch_params
        self.solver_params = solver_params

        self.learning_rate_func = common.create_lr_func(solver_params)

        # with tf.device('/cpu:0')

        self.model = CONTROLLER(self.simulator, self.arch_params, self.solver_params, params)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(self.init)

        self.sn_dir = sn_dir
        self.avg_cost = 999
        self.grad_mean = 0
        self.param_abs_norm = 0

    def train_step(self, iter):
        n_steps = self.arch_params['n_steps_train']
        time_vec = [float(t) for t in range(n_steps)]
        state_0 = self.simulator.drill_state()
        lr = self.learning_rate_func(iter,self.solver_params)

        returned_train_vals = self.sess.run(fetches=[self.model.apply_grads,
                                                     self.model.cost],
                                            feed_dict={self.model.state_0:state_0,
                                                       self.model.time_vec:time_vec,
                                                       self.model.lr:lr})

        self.avg_cost = 0.95 * self.avg_cost + 0.05 * returned_train_vals[1]

        if iter % 10 == 0:
            buf = "processing iter: %d, cost: (%0.2f)" % (iter, self.avg_cost)
            sys.stdout.write('\r' + buf)

    def test_step(self):
        n_steps = self.arch_params['n_steps_test']
        time_vec = [float(t) for t in range(n_steps)]
        state_0 = self.simulator.drill_state()
        returned_test_vals = self.sess.run(fetches=[self.model.scan_returns,
                                                    self.model.mean_abs_grad,
                                                    self.model.mean_abs_w
                                                    ],
                                           feed_dict={self.model.state_0:state_0,
                                                    self.model.time_vec: time_vec})

        self.simulator.returned_test_vals = returned_test_vals[0]
        self.grad_mean = returned_test_vals[1]
        self.param_abs_norm = returned_test_vals[2]

        self.simulator.state_0 = state_0

    def print_info_line(self, iter):
        buf = '%s Training ctrlr 1: iter %d, cost: %0.4f, grad_mean %0.12f, abs norm %0.6f, lr %0.6f\n' % \
                                    (datetime.datetime.now(), iter, self.avg_cost, self.grad_mean, self.param_abs_norm, self.learning_rate_func(iter, self.solver_params))
        sys.stdout.write('\r' + buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % iter), saver=self.saver, session=self.sess)
