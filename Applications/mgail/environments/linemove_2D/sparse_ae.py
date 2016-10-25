import tensorflow as tf
import common

class SPARSE_AE(object):

    def __init__(self, in_dim, hidden_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': in_dim,
            'n_hidden_0': hidden_dim,
        }

        self.solver_params = {
            'lr': 0.001,
            'weight_decay': 0.000001,
            'weights_stddev': 0.08,
        }

        self._init_layers(weights)

    def forward(self, input):
        '''
        state_: matrix
        action: matrix
        '''

        z0 = tf.nn.xw_plus_b(input, self.weights['0'], self.biases['0'], name='h0')
        h0 = tf.nn.tanh(z0)

        self.h1 = tf.nn.xw_plus_b(h0, self.weights['1'], self.biases['1'], name='h1')

        return h0

    def backward(self, loss):

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values() + self.biases.values())

        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def train(self, objective):
        self.loss = objective
        self.minimize, self.mean_abs_grad, self.mean_abs_w = self.backward(self.loss)

    def _init_layers(self, weights):

        # if a trained model is given
        if weights != None:
            print 'Loading weights... '

        # if no trained model is given
        else:
            weights = {
                '0': tf.Variable(tf.random_normal([self.arch_params['in_dim']      , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'])),
            }

            biases = {
                '0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                '1': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'])),
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()
