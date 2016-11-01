import tensorflow as tf
import common

class TRANSITION(object):

    def __init__(self, in_dim, out_dim, size, lr):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0], #800,
            'n_hidden_1': size[1], #400,
            # 'n_hidden_2': 100,
        }

        self.solver_params = {
            'lr': lr, #0.0001,
            'weight_decay': 0.000001,
            'weights_stddev': 0.08,
        }

        self._init_layers()

    def forward(self, state_, action, autoencoder):
        '''
        :param _input: N_batch x np.concatenate([[x_h, x_ct, x_h_, x_ct_, x_h__, x_ct__], v_t, x_t, a_t, self.is_aggressive, [ct_ind]])
        :param _input: N_batch x action
        :return: prediction: {x_h,x_ct}_t
        '''

        x_H_ = tf.slice(state_, [0, 0], [-1, 6])
        x_ = tf.slice(state_, [0, 0], [-1, 2])
        rest = tf.slice(state_, [0, 6], [-1, -1])

        _input = tf.concat(concat_dim=1, values=[x_H_, action], name='input')

        h0 = tf.nn.xw_plus_b(_input, self.weights['0'], self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        relu1 = tf.nn.relu(h1)

        delta = tf.nn.xw_plus_b(relu1, self.weights['c'], self.biases['c'], name='delta')

        x = x_ + delta

        x_H = tf.concat(concat_dim=1, values=[x, tf.slice(x_H_, [0, 0], [-1, 4])])

        state = tf.concat(concat_dim=1, values=[x_H, rest], name='state')

        return state

    def backward(self, loss):

        # create an optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])

        # compute the gradients for a list of variables
        grads_and_vars = self.opt.compute_gradients(loss=loss, var_list=self.trainable_variables)

        self.mean_abs_grad, self.mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        self.minimize = self.opt.apply_gradients(grads_and_vars)

    def train(self, objective):
        self.loss = objective
        self.backward(self.loss)
        self.loss_summary = tf.scalar_summary('loss_t', objective)

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
