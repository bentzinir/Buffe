import tensorflow as tf
import common

class CONTROLLER(object):

    def __init__(self, simulator, arch_params, solver_params, params):

        self.simulator = simulator

        self._init_layers(params, arch_params, solver_params)

        self.time_vec = tf.placeholder("float")
        self.lr = tf.placeholder("float")
        self.state_0 = tf.placeholder("float", shape=self.simulator.state_dim)

        def _recurrence(state_t_,time):

            _state_t_ = tf.slice(state_t_, [0], [2 + 3*self.simulator.n_t])

            _state_t_ = tf.expand_dims(_state_t_, 0)

            h0 = tf.add(tf.matmul(_state_t_, self.weights['w_0']), self.biases['b_0'],name='h0')
            relu0 = tf.nn.relu(h0)

            h1 = tf.add(tf.matmul(relu0, self.weights['w_1']), self.biases['b_1'],name='h1')
            relu1 = tf.nn.relu(h1)

            h2 = tf.mul(tf.matmul(relu1, self.weights['w_2']), self.biases['b_2'],name='h2')
            relu2 = tf.nn.relu(h2)

            a = tf.matmul(relu2, self.weights['w_c'],name='a')

            a = tf.squeeze(a)

            # a = tf.Print(a,[a],message='a: ')

            state_t = self.simulator.step(state_t_, a)

            step_costs = self.simulator.step_cost(state_t, a, time)

            return tf.concat(concat_dim=0, values=[state_t, step_costs], name='scan_return')

        self.scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.state_0)

        self.cost, self.cost_weighted = self.simulator.total_cost(self.scan_returns)

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(self.cost_weighted)

        # apply the gradient
        self.apply_grads = opt.apply_gradients(grads_and_vars)

        # output gradient and weights information
        self.mean_abs_grad, self.mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

    def _init_layers(self, params, arch_params, solver_params):

        # if a trained model is given
        if params != None:
            pass

        # if no trained model is given
        else:
            weights = {
                'w_0': tf.Variable(tf.random_normal([2 + 3*self.simulator.n_t, arch_params['n_hidden_0']], stddev=solver_params['weights_stddev'])),
                'w_1': tf.Variable(tf.random_normal([arch_params['n_hidden_0'], arch_params['n_hidden_1']], stddev=solver_params['weights_stddev'])),
                'w_2': tf.Variable(tf.random_normal([arch_params['n_hidden_1'], arch_params['n_hidden_2']], stddev=solver_params['weights_stddev'])),
                'w_c': tf.Variable(tf.random_normal([arch_params['n_hidden_2'], 1])),
            }

            biases = {
                'b_0': tf.Variable(tf.random_normal([arch_params['n_hidden_0']], stddev=solver_params['weights_stddev'])),
                'b_1': tf.Variable(tf.random_normal([arch_params['n_hidden_1']], stddev=solver_params['weights_stddev'])),
                'b_2': tf.Variable(tf.random_normal([arch_params['n_hidden_2']], stddev=solver_params['weights_stddev']))
            }
        self.weights = weights
        self.biases = biases