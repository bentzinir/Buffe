import tensorflow as tf
import common

class DISCRIMINATOR(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 250,
            'n_hidden_1': 250,
            # 'n_hidden_2': 50,
        }

        self.solver_params = {
            'lr': 0.001,
            # 'grad_clip_val': 5,
            'weight_decay': 0.000001,
            'weights_stddev': 0.1,
        }

        self._init_layers(weights)

    def forward(self, state_, action):
        '''
        state_: matrix
        action: matrix
        '''

        # clip observation variables from full state
        x_H_ = tf.slice(state_, [0, 0], [-1, 6])

        _input = tf.concat(concat_dim=1, values=[x_H_, action], name='input')

        h0 = tf.nn.xw_plus_b(_input, self.weights['0'], self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.nn.xw_plus_b(relu0, self.weights['1'], self.biases['1'], name='h1')
        relu1 = tf.nn.relu(h1)

        # h2 = tf.nn.xw_plus_b(relu1, self.weights['2'], self.biases['2'], name='h2')
        # relu2 = tf.nn.relu(h2)

        d = tf.nn.xw_plus_b(relu1, self.weights['c'], self.biases['c'], name='d')

        return d

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

    def zero_one_loss(self, state, action, label):
        d = self.forward(state, action)
        labels = tf.concat(1, [1 - label, label])
        predictions = tf.argmax(input=d, dimension=1)
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def train(self, state, action, label):
        # 1. calculate 0-1 accuracy
        self.zero_one_loss(state, action, label)

        # 2. calculate loss and compute gradients
        d = self.forward(state, action)
        labels = tf.concat(1, [1-label, label])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels))
        self.minimize = self.backward(self.loss)

        s_grad = tf.gradients(self.loss, state)[0]
        s_grad = tf.reduce_mean(tf.abs(s_grad), 0)
        s_grad = tf.slice(s_grad, [0], [6])
        a_grad = tf.gradients(self.loss, action)[0]
        a_grad = tf.reduce_mean(tf.abs(a_grad), 0)

        self.state_action_grad = tf.concat(concat_dim=0, values=[s_grad, a_grad])

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
                #'2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'])),
                'c': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1'], self.arch_params['out_dim']]   , stddev=self.solver_params['weights_stddev'])),
            }

            biases = {
                '0': tf.Variable(tf.random_normal([self.arch_params['n_hidden_0']], stddev=self.solver_params['weights_stddev'], mean=0.0)),
                '1': tf.Variable(tf.random_normal([self.arch_params['n_hidden_1']], stddev=self.solver_params['weights_stddev'], mean=0.0)),
                #'2': tf.Variable(tf.random_normal([self.arch_params['n_hidden_2']], stddev=self.solver_params['weights_stddev'],mean=0.0)),
                'c': tf.Variable(tf.random_normal([self.arch_params['out_dim']], stddev=self.solver_params['weights_stddev'],    mean=0.0)),
            }
        self.weights = weights
        self.biases = biases
        self.trainable_variables = weights.values() + biases.values()
