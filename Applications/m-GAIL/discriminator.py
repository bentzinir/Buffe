import tensorflow as tf
import common

class DISCRIMINATOR(object):

    def __init__(self, in_dim, out_dim, weights=None):

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': 300,
            'n_hidden_1': 400,
            'n_hidden_2': 300,
            'n_hidden_3': 200
        }

        self.solver_params = {
            # 'lr_type': 'episodic', 'base': 0.001, 'interval': 5e3,
            # 'lr_type': 'inv', 'base': 0.00005, 'gamma': 0.0001, 'power': 0.75,
            # 'lr_type': 'fixed', 'base': 0.003,
            'lr': 0.001,
            # 'grad_clip_val': 5,
            'weight_decay': 0.000001,
            'weights_stddev': 0.3,
        }

        self._init_layers(weights)

    def forward(self, state, action):
        '''
        state_: matrix
        action: vector
        '''

        action = tf.expand_dims(action, 1)

        _input = tf.concat(concat_dim=1, values=[state, action], name='input')

        h0 = tf.add(tf.matmul(_input, self.weights['0']), self.biases['0'], name='h0')
        relu0 = tf.nn.relu(h0)

        h1 = tf.add(tf.matmul(relu0, self.weights['1']), self.biases['1'], name='h1')
        relu1 = tf.nn.relu(h1)

        prediction = tf.add(tf.matmul(relu1, self.weights['c']), self.biases['c'], name='prediction')

        # prediction = tf.mul(0., prediction) + 0.5
        prediction = tf.Print(prediction, [prediction], message='prediction:')

        return prediction

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

    def zero_one_loss(self, state, action, state_e, action_e):
        d_agent = self.forward(state, action)
        d_expert = self.forward(state_e, action_e)
        acc_agent = tf.reduce_mean(tf.to_float(d_agent < 0.5))
        acc_expert = tf.reduce_mean(tf.to_float(d_expert > 0.5))

        #acc_agent = tf.Print(acc_agent, [acc_agent], message='acc_agent:')
        #acc_expert = tf.Print(acc_expert, [acc_expert], message='acc_expert:')

        self.acc = tf.div(acc_agent+acc_expert, 2)


    def train(self, state, action, label):

        # d_agent = self.forward(state, action)
        # d_expert = self.forward(state_e, action_e)
        #
        # # clip predictions
        # d_agent = tf.clip_by_value(d_agent, 0.0, 1)
        # d_expert = tf.clip_by_value(d_expert, 0.0, 1)
        #
        # # discriminator's goal is to maximize the following expression:
        # # self.loss = -(tf.reduce_mean(tf.log(d_agent)) + tf.reduce_mean(tf.log(1-d_expert)))
        # self.loss = (tf.reduce_mean(tf.abs(1-d_agent)) + tf.reduce_mean(tf.abs(d_expert)))

        prediction = self.forward(state, action)
        self.loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.mul(label, prediction))))

        self.acc = self.loss

        #loss_agent = tf.reduce_mean(tf.log(d_agent))
        #loss_expert = tf.reduce_mean(tf.log(1-d_expert))

        #loss_agent = tf.Print(loss_agent, [loss_agent], message='loss_agent')
        #loss_expert = tf.Print(loss_expert, [loss_expert], message='loss_expert')

        #self.loss = -(loss_agent + loss_expert)

        self.minimize = self.backward(self.loss)

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
