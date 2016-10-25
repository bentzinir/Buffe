import tensorflow as tf
import common

class DISCRIMINATOR(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 50, # 20
            'n_hidden_1': 100, # 20
        }

        self.solver_params = {
            'lr': 0.001,
            # 'grad_clip_val': 5,
            'weight_decay': 0.000001,
            'weights_stddev': 0.1,
        }

        self._init_layers(weights)

    def forward(self, state, action, autoencoder):
        '''
        state_: matrix
        action: matrix
        '''

        if autoencoder is None:
            _input = state
        else:
            _input = autoencoder.forward(state)

        concat = tf.concat(concat_dim=1, values=[_input, action], name='input')

        h0 = tf.nn.xw_plus_b(concat, self.weights['0'], self.biases['0'], name='h0')
        # relu0 = tf.nn.relu(h0)
        relu0 = common.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        # relu1 = tf.nn.relu(h1)
        relu1 = common.relu(h1)

        relu1_do = tf.nn.dropout(relu1, 0.4)

        d = tf.nn.xw_plus_b(relu1_do, self.weights['c'], self.biases['c'], name='d')

        return d

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
        self.loss_summary = tf.scalar_summary('loss_d', objective)

    def _init_layers(self, weights):
        # if a trained model is given
        if weights != None:
            print 'Loading weights... '
            biases = None

        # if no trained model is given
        else:
            weights = {
                '0': tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                'c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev'])),
            }

            biases = {
                '0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'], mean=0.0)),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'], mean=0.0)),
                'c': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'],    mean=0.0)),
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()
