import tensorflow as tf
import common

class TRANSITION_SEP(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 50,
            'n_hidden_1': 100,
        }

        self.solver_params = {
            'lr': 0.0001,
            # 'grad_clip_val': 5,
            'weight_decay': 0.000001,
            'weights_stddev': 0.15,
        }

        self._init_layers(weights)

    def forward(self, state_, action):

        input = tf.concat(concat_dim=3, values=[state_, action], name='input')

        input_tiled = tf.tile(input, [1, 1, self.arch_params['out_dim'], 1])

        h0 = tf.nn.relu(self.conv2d(input_tiled, self.weights['0']) + self.biases['0'])

        # h1 = tf.nn.relu(self.conv2d(h0, self.weights['1']) + self.biases['1'])
        #
        # h2 = tf.nn.relu(self.conv2d(h1, self.weights['2']) + self.biases['2'])

        h3 = self.conv2d(h0, self.weights['3']) + self.biases['3']

        delta = tf.squeeze(self.conv2d(h3, self.weights['c'])) + self.biases['c']

        state = state_ + delta

        # state = tf.Print(state, [state], message='state: ', summarize=17)

        return state

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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

    def train(self, state_, action, state):
        state_a = self.forward(state_, action)
        self.loss = tf.nn.l2_loss(state_a-state)
        self.minimize = self.backward(self.loss)

    def _init_layers(self, weights):

        # if a trained model is given
        if weights != None:
            print 'Loading weights... '

        # if no trained model is given
        else:
            weights = {
                'w_0': tf.Variable(tf.random_normal([1,1,self.arch_params['in_dim'], self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'w_1': tf.Variable(tf.random_normal([1,1,self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                'w_c': tf.Variable(tf.random_normal([1,1,self.arch_params['n_hidden_1'], 1]   , stddev=self.solver_params['weights_stddev'])),
            }

            biases = {
                'b_0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'])),
                'b_1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'])),
                'b_c': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev']))
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()