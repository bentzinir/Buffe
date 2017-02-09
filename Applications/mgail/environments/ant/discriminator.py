import tensorflow as tf
import common
from collections import OrderedDict

class DISCRIMINATOR(object):

    def __init__(self, in_dim, out_dim, size, lr, do_keep_prob, weight_decay):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'n_hidden_2': size[2],
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weights_stddev': 0.1,
        }

        self._init_layers()

    def forward(self, state, action, autoencoder):
        '''
        state_: matrix
        action: matrix
        '''

        if autoencoder is None:
            _input = state
        else:
            _input, _ = autoencoder.forward(state)

        concat = tf.concat(concat_dim=1, values=[_input, action], name='input')

        h0 = tf.nn.xw_plus_b(concat, self.weights['0'], self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)

        # h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        # relu1 = tf.nn.relu(h1)
        #
        # h2 = tf.nn.xw_plus_b(relu1, self.weights['2'], self.biases['2'], name='h2')
        # relu2 = tf.nn.relu(h2)

        relu1_do = tf.nn.dropout(relu0, self.arch_params['do_keep_prob'])

        d = tf.nn.xw_plus_b(relu1_do, self.weights['c'], self.biases['c'], name='d')

        return d

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

    def train(self, objective):
        self.loss = objective
        self.minimize, self.mean_abs_grad, self.mean_abs_w = self.backward(self.loss)
        self.loss_summary = tf.scalar_summary('loss_d', objective)

    def create_variables(self):
        weights = OrderedDict([
            ('0', tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev']))),
            # ('1', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev']))),
            # ('2', tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev']))),
            ('c', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev']))),
        ])

        biases = OrderedDict([
            ('0', tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev']))),
            # ('1', tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev']))),
            # ('2', tf.Variable(tf.random_normal([self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev']))),
            ('c', tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'])))
        ])
        return weights, biases

    def _init_layers(self):
        self.weights, self.biases = self.create_variables()
