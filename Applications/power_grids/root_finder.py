import tensorflow as tf
import sys
sys.path.append('/Users/galdalal/PycharmProjects/Buffe/utils')
import common
from collections import OrderedDict

class ROOT_FINDER(object):

    def __init__(self, in_dim, out_dim, size, do_keep_prob, lr=0.001, w_std=0.08, weight_decay=1e-7):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weights_stddev': w_std,
        }

        self._init_layers()

    def forward(self, params):

        h0 = tf.nn.xw_plus_b(params, self.weights['w0'], self.biases['b0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['w1'], self.biases['b1'], name='h1')
        relu1 = tf.nn.relu(h1)

        relu1_do = tf.nn.dropout(relu1, self.arch_params['do_keep_prob'])

        roots = tf.nn.xw_plus_b(relu1_do, self.weights['wc'], self.biases['bc'], name='a')

        return roots

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.weights.values()])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values() + self.biases.values())

        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def create_variables(self):
        weights = OrderedDict([
            ('w0', tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev']))),
            ('w1', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev']))),
            ('wc', tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev']))),
        ])

        biases = OrderedDict([
            ('b0', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev']))),
            ('b1', tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev']))),
            ('bc', tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'])))
        ])
        return weights, biases

    def _init_layers(self):
        self.weights, self.biases = self.create_variables()

        weights, biases = self.create_variables()
        self.accum_grads = weights.copy()
        self.accum_grads.update(biases)

        self.reset_grad_op = []
        for acc_grad in self.accum_grads.values():
            self.reset_grad_op.append(acc_grad.assign(0. * acc_grad))

    def copy_weights(self, weights, biases):
        self.copy_weights_op = []
        for key, value in self.weights.iteritems():
            self.copy_weights_op.append(value.assign(weights[key]))
        for key, value in self.biases.iteritems():
            self.copy_weights_op.append(value.assign(biases[key]))
