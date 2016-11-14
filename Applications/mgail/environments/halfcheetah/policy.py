import tensorflow as tf
import common

class POLICY(object):

    def __init__(self, in_dim, out_dim, size, lr, w_std, do_keep_prob):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0], #150,
            'n_hidden_1': size[1], #75,
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr, #0.001,
            'weight_decay': 0.000001,
            'weights_stddev': w_std,
        }

        self._init_layers()

    def forward(self, state, autoencoder):
        '''
        state: vector
        '''

        if autoencoder is None:
            _input = state
        else:
            _input, _ = autoencoder.forward(state)

        h0 = tf.nn.xw_plus_b(_input, self.weights['0'], self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)
        # relu0 = common.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        relu1 = tf.nn.relu(h1)
        # relu1 = common.relu(h1)

        relu1_do = tf.nn.dropout(relu1, self.arch_params['do_keep_prob'])

        a = tf.nn.xw_plus_b(relu1_do, self.weights['c'], self.biases['c'], name='a')

        return a

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.trainable_variables)

        grads_and_vars = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in grads_and_vars]

        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def train(self, objective, mode):
        setattr(self, 'loss_' + mode, objective)
        backward = self.backward(getattr(self, 'loss_' + mode))
        setattr(self, 'minimize_' + mode, backward[0])
        setattr(self, 'mean_abs_grad_' + mode, backward[1])
        setattr(self, 'mean_abs_w_' + mode, backward[2])

    def _init_layers(self):
        weights = {
            '0': tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
            '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
            'c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev'])),
        }

        biases = {
            '0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
            '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
            'c': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev']))
        }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()
