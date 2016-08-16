import tensorflow as tf
import common

class POLICY(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 50,
            'n_hidden_1': 50,
            'n_hidden_2': 50,
        }

        self.solver_params = {
            # 'lr_type': 'episodic', 'base': 0.001, 'interval': 5e3,
            # 'lr_type': 'inv', 'base': 0.0001, 'gamma': 0.0001, 'power': 0.75,
            # 'lr_type': 'fixed', 'base': 0.003,
            'lr': 0.001,
            # 'grad_clip_val': 5,
            'weight_decay': 0.000001,
            'weights_stddev': 0.15,
            'max_importance_weight': 5.,
        }

        self._init_layers(weights)

    def forward(self, state):
        '''
        state: vector
        '''

        # state = tf.expand_dims(state, 0)

        h0 = tf.nn.xw_plus_b(state, self.weights['0'], self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        relu1 = tf.nn.relu(h1)

        # h2 = tf.nn.xw_plus_b(relu1, self.weights['2'], self.biases['2'], name='h2')
        # relu2 = tf.nn.relu(h2)

        a = tf.nn.xw_plus_b(relu1, self.weights['c'], self.biases['c'], name='a')

        return a

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])

        # compute the gradients for a list of variables
        self.grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.trainable_variables)

        self.grads_and_vars = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in self.grads_and_vars]

        self.mean_abs_grad, self.mean_abs_w = common.compute_mean_abs_norm(self.grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(self.grads_and_vars)

        return apply_grads

    def train(self, policy_obj):
        self.loss = policy_obj
        self.minimize = self.backward(self.loss)

    def _init_layers(self, weights):

        # if a trained model is given
        if weights is not None:
            print 'Loading weights... '

        # if no trained model is given
        else:
            weights = {
                '0': tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                '2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_2'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev'])),
            }

            biases = {
                '0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                '2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'c': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev']))
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()
