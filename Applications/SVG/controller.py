import numpy as np
import tensorflow as tf
import common

class CONTROLLER(object):

    def __init__(self, state_0, time_vec, lr, game_params, arch_params, solver_params, params):

        self._init_layers(params, arch_params, game_params)

        self._connect(game_params, solver_params)

        def _step_state(self, state_t_, a):

            # 0. slice previous state
            v_h_ = tf.slice(state_t_,          [0],              [1])
            x_h_ = tf.slice(state_t_,          [1],              [1])
            v_t_ = tf.slice(state_t_,          [2],              [self.n_t])
            x_t_ = tf.slice(state_t_,          [2 + 1*self.n_t], [self.n_t])
            a_t_ = tf.slice(state_t_,          [2 + 2*self.n_t], [self.n_t])
            is_aggressive = tf.slice(state_t_, [2 + 3*self.n_t], [self.n_t])

            # 1. simulator logic
            next_x_t_ = tf.concat(concat_dim=0, values=[tf.slice(x_t_, [1], [self.n_t-1]), tf.slice(x_t_, [0], [1])], name='state')

            relx = next_x_t_ - x_t_

            # x_t_ = tf.Print(x_t_, [x_t_], message='x_t_:', summarize=self.n_t)

            # fix the jump between -pi and +pi
            relx = tf.mul(tf.to_float(relx>=0),relx) + tf.mul(tf.to_float(relx<0),self.two_pi_r + relx)

            relx_to_host = x_h_ - x_t_

            is_host_cipv = tf.mul(tf.mul(tf.to_float((x_h_>-0.5*self.host_length)),tf.to_float(x_h_>x_t_)),tf.to_float(relx_to_host<relx))

            # # If host CIPV - Change relx to him
            relx = tf.mul(is_host_cipv,tf.mul(tf.to_float(x_h_>0),tf.to_float(x_h_-x_t_)))+tf.mul((1-is_host_cipv),relx)

            is_host_approaching = tf.mul(tf.mul(tf.mul(tf.mul(tf.to_float(x_h_> -1.5*self.host_length),tf.to_float(x_h_ <= -0.5*self.host_length)),tf.to_float(x_t_<0)),tf.to_float(x_t_ > -0.25*self.two_pi_r)),tf.to_float(tf.logical_or(next_x_t_>0,next_x_t_<x_t_)))

            accel_default = tf.mul(5.,relx) - tf.mul(10.,v_t_)

            accel_is_aggressive = tf.maximum(3.,tf.mul(3.,relx_to_host - tf.mul(1.5,v_t_)))

            accel_not_aggressive = tf.div(0.5-v_t_,self.dt)

            accel_host_approaching = tf.mul(is_aggressive,accel_is_aggressive) + tf.mul(1-is_aggressive,accel_not_aggressive)

            accel = tf.mul(is_host_approaching,accel_host_approaching) + tf.mul(1-is_host_approaching,accel_default)

            # 2. exact next state
            # 2.1 host
            v_h = v_h_ + tf.mul(self.dt, a)

            # clip host speed to the section [0,v0]
            v_h = tf.clip_by_value(v_h, 0, self.v_0)

            x_h = x_h_ + self.dt * v_h

            x_h = tf.mul(tf.to_float(x_h >= 0.5*self.two_pi_r),x_h - self.two_pi_r) + tf.mul(tf.to_float(x_h < 0.5*self.two_pi_r),x_h)

            # 2.2 targets
            v_t_e = tf.maximum(0., v_t_ + tf.mul(self.dt, accel))
            x_t_e = x_t_ + tf.mul(self.dt,v_t_e)
            a_t_e = v_t_e - v_t_

            # 3. learn the transition model between states
            v_t_a = v_t_e
            x_t_a = x_t_e
            a_t_a = a_t_e

            # 4. prediction noise
            n_v_t = v_t_e - v_t_a
            n_x_t = x_t_e - x_t_a
            n_a_t = a_t_e - a_t_a

            # 5. disconnect the gradient of the noise signals
            tf.stop_gradient(n_v_t)
            tf.stop_gradient(n_x_t)
            tf.stop_gradient(n_a_t)

            # 6. add the noise to the approximation
            v_t = v_t_a + n_v_t
            x_t = x_t_a + n_x_t
            a_t = a_t_a + n_a_t

            # apply [-pi,pi] discontinuity
            x_t = tf.mul(tf.to_float(x_t >= 0.5*self.two_pi_r),x_t-self.two_pi_r) + tf.mul(x_t,tf.to_float(x_t<0.5*self.two_pi_r))

            state_t = tf.concat(concat_dim=0, values=[v_h,x_h,v_t,x_t,a_t], name='state')

            return state_t

        def _recurrence(state_t_,time):

            _state_t_ = tf.slice(state_t_, [0], [2 + 3*self.n_t])

            _state_t_ = tf.expand_dims(_state_t_, 0)

            h0 = tf.add(tf.matmul(_state_t_, self.weights['w_0']), self.biases['b_0'],name='h0')
            relu0 = tf.nn.relu(h0)

            h1 = tf.add(tf.matmul(relu0, self.weights['w_1']), self.biases['b_1'],name='h1')
            relu1 = tf.nn.relu(h1)

            h2 = tf.mul(tf.matmul(relu1, self.weights['w_2']), self.biases['b_2'],name='h2')
            relu2 = tf.nn.relu(h2)

            a = tf.matmul(relu2, self.weights['w_c'],name='a')

            a = tf.squeeze(a)

            state_t = _step_state(self, state_t_, a)

            # cost:
            x_h = tf.slice(state_t, [1], [1])
            x_t = tf.slice(state_t, [2 + self.n_t], [self.n_t])
            is_aggressive = tf.slice(state_t_, [2 + 3 * self.n_t], [self.n_t])

            # 0. smooth acceleration policy
            cost_accel = tf.abs(a)
            cost_accel_d = tf.mul(tf.pow(self.gamma, time), cost_accel)
            cost_accel_d = tf.expand_dims(cost_accel_d, 0)

            # 1. forcing the host to move forward (until the top point of the roundabout)
            # cost_progress = tf.nn.relu(0.5 * self.two_pi_r - x_h)
            cost_prog = tf.square(0.25 * self.two_pi_r - x_h)
            cost_prog_d = tf.mul(tf.pow(self.gamma, time), cost_prog)

            # 2. keeping distance from close vehicles
            x_abs_diffs = tf.abs(x_h - x_t)

            cost_acci = tf.reduce_mean(tf.mul(tf.nn.relu(self.require_distance - x_abs_diffs),3.))
            cost_acci_d = tf.mul(tf.pow(self.gamma, time), cost_acci)
            cost_acci_d = tf.expand_dims(cost_acci_d, 0)

            # return tf.concat(concat_dim=1, values=[state_t, is_aggressive, cost_accel_d, cost_prog_d, cost_acci_d], name='scan_return')
            return tf.concat(concat_dim=0, values=[state_t, is_aggressive, cost_accel_d, cost_prog_d], name='scan_return')

        scan_returns = tf.scan(fn=_recurrence, elems=time_vec, initializer=state_0)

        # slice different cost measures
        costs_accel = tf.slice(scan_returns, [0, 2 + 4 * self.n_t], [-1, 1])
        costs_prog = tf.slice(scan_returns, [0, 2 + 4 * self.n_t + 1], [-1, 1])
        # costs_acci = tf.slice(scan_returns, [0, 2 + 4 * self.n_t + 2], [-1, 1])

        # average over time
        cost_accel = tf.reduce_mean(costs_accel)
        cost_prog = tf.reduce_mean(costs_prog)
        # cost_acci = tf.reduce_mean(costs_acci)

        # for printings
        # self.cost = cost_accel + cost_prog # + cost_acci
        self.cost = cost_prog

        # for training
        self.cost_weighted = tf.mul(cost_accel, self.alpha_accel) + tf.mul(cost_prog, self.alpha_progress) # + tf.mul(cost_acci, self.alpha_accident)

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(self.cost_weighted)

        # apply the gradient
        self.apply_grads = opt.apply_gradients(grads_and_vars)

        self.mean_abs_grad, self.mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)


        self.v_h = tf.slice(scan_returns, [0, 0],               [-1, 1])
        self.x_h = tf.slice(scan_returns, [0, 1],               [-1, 1])
        self.v_t = tf.slice(scan_returns, [0, 2],               [-1, self.n_t])
        self.x_t = tf.slice(scan_returns, [0, 2 + self.n_t],    [-1, self.n_t])

    def _connect(self, game_params, solver_params):

        self.dt = game_params['dt']
        self.alpha_accel = game_params['alpha_accel']
        self.alpha_progress = game_params['alpha_progress']
        self.alpha_accident = game_params['alpha_accident']
        self.two_pi_r = 2 * np.pi * game_params['r']
        self.num_targets = game_params['num_of_targets']
        self.require_distance = game_params['require_distance']
        self.v_0 = game_params['v_0']
        self.r = game_params['r']
        self.host_length = game_params['host_length']
        self.alpha_cost_a = game_params['alpha_cost_a']
        self.alpha_cost_t = game_params['alpha_cost_t']
        self.n_t = game_params['num_of_targets']
        self.gamma = game_params['gamma']
        self.state_dim = game_params['state_dim']

        self.l1_weight = solver_params['l1_weight_decay']

    def _init_layers(self, params, arch_params, game_params):

        # if a trained model is given
        if params != None:
            pass

        # if no trained model is given
        else:
            weights = {
                'w_0': tf.Variable(tf.random_normal([2 + 3*game_params['num_of_targets'], arch_params['n_hidden_0']])),
                'w_1': tf.Variable(tf.random_normal([arch_params['n_hidden_0'], arch_params['n_hidden_1']])),
                'w_2': tf.Variable(tf.random_normal([arch_params['n_hidden_1'], arch_params['n_hidden_2']])),
                'w_c': tf.Variable(tf.random_normal([arch_params['n_hidden_2'], 1])),
            }

            biases = {
                'b_0': tf.Variable(tf.random_normal([arch_params['n_hidden_0']])),
                'b_1': tf.Variable(tf.random_normal([arch_params['n_hidden_1']])),
                'b_2': tf.Variable(tf.random_normal([arch_params['n_hidden_2']]))
            }
        self.weights = weights
        self.biases = biases
        self.params = weights.values() + biases.values()
