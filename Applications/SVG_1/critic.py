import tensorflow as tf
import common

class CRITIC(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 400,
            'n_hidden_1': 200,
            # 'n_hidden_2': 100,
            # 'n_hidden_3': 100
        }

        self.solver_params = {
            # 'lr_type': 'episodic', 'base': 0.001, 'interval': 5e3,
            # 'lr_type': 'inv', 'base': 0.00005, 'gamma': 0.0001, 'power': 0.75,
            # 'lr_type': 'fixed', 'base': 0.003,
            'lr': 0.00001,
            'weight_decay': 0.000001,
            'weights_stddev': 0.15,
            'target_update_freq': 5000,
        }

        self._init_layers(weights)

    def forward(self, state):

        h0 = tf.add(tf.matmul(state, self.weights['w_0']), self.biases['b_0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.add(tf.matmul(relu0, self.weights['w_1']), self.biases['b_1'], name='h1')
        relu1 = tf.nn.relu(h1)

        # h2 = tf.add(tf.matmul(relu1, self.weights['w_2']), self.biases['b_2'], name='h2')
        # relu2 = tf.nn.relu(h2)

        value = tf.mul(tf.matmul(relu1, self.weights['w_c']), self.biases['b_c'], name='prediction')

        return value

    def forward_t(self, state):

        h0 = tf.add(tf.matmul(state, self.weights_t['w_0']), self.biases_t['b_0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.add(tf.matmul(relu0, self.weights_t['w_1']), self.biases_t['b_1'], name='h1')
        relu1 = tf.nn.relu(h1)

        # h2 = tf.add(tf.matmul(relu1, self.weights_t['w_2']), self.biases_t['b_2'], name='h2')
        # relu2 = tf.nn.relu(h2)

        value = tf.mul(tf.matmul(relu1, self.weights_t['w_c']), self.biases_t['b_c'], name='prediction')

        return value

    def backward(self, loss):

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        if self.solver_params['weight_decay']:
            loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])

        # compute the gradients for a list of variables
        self.grads_and_vars = opt.compute_gradients(loss=loss,
                                                    var_list=self.weights.values() + self.biases.values()
                                                    )

        self.mean_abs_grad, self.mean_abs_w = common.compute_mean_abs_norm(self.grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(self.grads_and_vars)

        return apply_grads

    def train(self, state_, state, r, gamma, w):
        v_ = self.forward(state_)
        v = self.forward_t(state)
        v = tf.stop_gradient(v)
        y = r + tf.mul(gamma, v)
        self.loss = tf.reduce_mean(tf.mul(w, tf.squeeze(tf.square(y-v_))))
        self.minimize = self.backward(self.loss)

    def _init_layers(self, weights):

        # if a trained model is given
        if weights is not None:
            print 'Loading weights... '

        # if no trained model is given
        else:
            weights = {
                'w_0': tf.Variable(tf.random_normal([self.arch_params['in_dim']    , self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'w_1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                # 'w_2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'w_c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev'])),
            }
            biases = {
                'b_0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'b_1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                # 'b_2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'b_c': tf.Variable(tf.random_normal([self.arch_params['out_dim']],    stddev=self.solver_params['weights_stddev']))
            }
            weights_t = {
                'w_0': tf.Variable(tf.random_normal([self.arch_params['in_dim'],     self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'w_1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                # 'w_2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'w_c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']],    stddev=self.solver_params['weights_stddev'])),
            }
            biases_t = {
                'b_0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'b_1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                # 'b_2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'b_c': tf.Variable(tf.random_normal([self.arch_params['out_dim']],    stddev=self.solver_params['weights_stddev']))
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()

        # target network
        self.weights_t = weights_t
        self.biases_t = biases_t

        self.copy_target_weights_op = []
        for key in self.weights:
            source = getattr(self, 'weights')[key]
            target = getattr(self, 'weights_t')[key]
            self.copy_target_weights_op.append(target.assign(source))

        for key in self.biases:
            source = getattr(self, 'biases')[key]
            target = getattr(self, 'biases_t')[key]
            self.copy_target_weights_op.append(target.assign(source))
