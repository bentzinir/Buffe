import math
import theano as t
import theano.tensor as tt
from theano import printing
import numpy as np
import common
import optimizers

class CONTROLLER(object):

    def __init__(self, v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, is_aggressive, lr, n_steps, game_params, arch_params, solver_params, params):

        self._init_layers(params, arch_params, game_params)

        self._connect(game_params, solver_params)

        def _step_state(v_h_, x_h_, v_t_, x_t_, a_t_, a, is_aggressive):

            next_x_t_ = tt.roll(x_t_,-1)

            relx = next_x_t_ - x_t_

            # fix the jump between -pi and +pi
            relx = (relx>=0) *relx + (relx<0)*(self.two_pi_r + relx)

            relx_to_host = x_h_ - x_t_

            is_host_cipv = (x_h_ > -0.5*self.host_length) * (x_h_ > x_t_) * (relx_to_host < relx)

            # If host CIPV - Change relx to him
            relx = (is_host_cipv * ((x_h_ > 0)*x_h_ - x_t_)) + ((1-is_host_cipv) * relx)

            is_host_approaching = (x_h_ > -1.5*self.host_length) * (x_h_ <= -0.5 *self.host_length) * (x_t_ < 0) * (x_t_ > -0.25*self.two_pi_r) * ((next_x_t_ > 0) + (next_x_t_ < x_t_))

            accel_default = 5*(relx - 2*v_t_)

            # accel_is_aggressive = (3 - v_t_)/self.dt
            accel_is_aggressive = tt.maximum(3, 3*(relx_to_host - 1.5*v_t_))

            accel_not_aggressive = (0.5 - v_t_)/self.dt

            accel_host_approaching = is_aggressive * accel_is_aggressive + (1 - is_aggressive) * accel_not_aggressive

            accel = is_host_approaching * accel_host_approaching + (1- is_host_approaching) * accel_default

            #1. exact next state

            #1.1 host
            v_h = v_h_ + self.dt * a

            # clip host speed to the section [0,v0]
            v_h = tt.clip(v_h, 0, 3*self.v_0)

            x_h = x_h_ + self.dt * v_h

            x_h = (x_h>=(self.two_pi_r/2)) * (x_h - self.two_pi_r) + (x_h < (self.two_pi_r/2)) * x_h

            #1.2 targets
            v_t_e = tt.maximum(0, v_t_ + self.dt * accel)

            x_t_e = x_t_ + self.dt * v_t_e

            a_t_e = v_t_e - v_t_

            #2. learn the transition model between states
            state_      = tt.concatenate([v_h_ , x_h_, tt.flatten(v_t_), tt.flatten(x_t_), tt.flatten(a_t_), a])
            state_t_e   = tt.concatenate([tt.flatten(v_t_e), tt.flatten(x_t_e), tt.flatten(a_t_e)])

            state_ = common.disconnected_grad(state_)
            state_t_e = common.disconnected_grad(state_t_e)

            h0 = tt.dot(state_, self.W_t_0) + self.b_t_0
            relu0 = tt.nnet.relu(h0)

            h1 = tt.dot(relu0, self.W_t_1) + self.b_t_1
            relu1 = tt.nnet.relu(h1)

            h2 = tt.dot(relu1, self.W_t_2) + self.b_t_2
            relu2 = tt.nnet.relu(h2)

            state_t_hat = tt.dot(relu2, self.W_t_c)

            cost_transition = tt.mean(tt.abs_(state_t_hat - state_t_e))

            v_t_a = (state_t_hat[0 : self.n_t]).dimshuffle(0,'x')
            x_t_a = (state_t_hat[self.n_t : 2 * self.n_t]).dimshuffle(0,'x')
            a_t_a = (state_t_hat[2 * self.n_t : ]).dimshuffle(0,'x')

            #3. prediction noise
            n_v_t = v_t_e - v_t_a
            n_x_t = x_t_e - x_t_a
            n_a_t = a_t_e - a_t_a

            #4. disconnect the gradient of the noise signals
            n_v_t = common.disconnected_grad(n_v_t)
            n_x_t = common.disconnected_grad(n_x_t)
            n_a_t = common.disconnected_grad(n_a_t)

            #5. add the noise to the approximation
            v_t = v_t_a + n_v_t
            x_t = x_t_a + n_x_t
            a_t = a_t_a + n_a_t

            # apply [-pi,pi] discontinuity
            x_t = (x_t>=(self.two_pi_r/2)) * (x_t - self.two_pi_r) + (x_t < (self.two_pi_r/2)) * x_t

            return v_h, x_h, v_t, x_t, a_t, cost_transition

        def _recurrence(v_h_, x_h_, v_t_, x_t_, a_t_, is_aggressive):

            state = tt.concatenate([v_h_, x_h_, tt.flatten(v_t_), tt.flatten(x_t_), tt.flatten(a_t_)])

            h0 = tt.dot(state, self.W_a_0) + self.b_a_0
            relu0 = tt.nnet.relu(h0)

            h1 = tt.dot(relu0, self.W_a_1) + self.b_a_1
            relu1 = tt.nnet.relu(h1)

            h2 = tt.dot(relu1, self.W_a_2) + self.b_a_2
            relu2 = tt.nnet.relu(h2)

            a = tt.dot(relu2, self.W_a_c)

            v_h, x_h, v_t, x_t, a_t, cost_transition = _step_state(v_h_, x_h_, v_t_, x_t_, a_t_, a, is_aggressive)

            # cost:

            # 0. smooth acceleration policy
            cost_accel = tt.abs_(a)

            # 1. forcing the host to move forward (until the top point of the roundabout)
            cost_progress = tt.nnet.relu(0.5*self.two_pi_r-x_h)

            # 2. keeping distance from close vehicles
            x_abs_diffs = tt.abs_(x_h - x_t)

            cost_accident =  tt.mean(3*tt.nnet.relu( self.require_distance-x_abs_diffs )) * tt.nnet.sigmoid(x_h - 0.5*self.host_length)

            cost = self.alpha_accel * cost_accel + self.alpha_progress * cost_progress + self.alpha_accident * cost_accident

            return (v_h, x_h, v_t, x_t, a_t, cost, cost_transition), t.scan_module.until(x_h[0]>=0.45*self.two_pi_r)

        [v_h, x_h, v_t, x_t, a_t, costs, costs_transition], scan_updates = t.scan(fn=_recurrence,
                                                sequences=None,
                                                outputs_info=[v_h_0, x_h_0, v_t_0, x_t_0, a_t_0, None, None],
                                                non_sequences=[is_aggressive],
                                                n_steps=n_steps,
                                                name='scan_func')

        self.cost_a = tt.mean(costs)
        self.cost_t = tt.mean(costs_transition)
        self.cost = self.alpha_cost_a * self.cost_a  + self.alpha_cost_t * self.cost_t

        obj_decayed = common.weight_decay(objective=self.cost, params=self.params, l1_weight=self.l1_weight)

        # obj_decayed_clip_grad = t.gradient.grad_clip(obj_decayed, -self.grad_clip_val, self.grad_clip_val)

        gradients = tt.grad(obj_decayed, self.params)

        self.updates = optimizers.optimizer(lr=lr, model=self, gradients=gradients, solver_params=solver_params)

        self.x_h = x_h
        self.v_h = v_h
        self.x_t = x_t
        self.v_t = v_t

        self.grad_mean = 0
        for g in gradients:
            self.grad_mean += tt.mean(tt.abs_(g))

        self.params_abs_norm = self._calc_params_norm()

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

        self.l1_weight = solver_params['l1_weight_decay']
        self.params = []
        # agent params
        self.params += [self.W_a_0, self.b_a_0]
        self.params += [self.W_a_1, self.b_a_1]
        self.params += [self.W_a_2, self.b_a_2]
        self.params += [self.W_a_c]
        # transition model params
        self.params += [self.W_t_0, self.b_t_0]
        self.params += [self.W_t_1, self.b_t_1]
        self.params += [self.W_t_2, self.b_t_2]
        self.params += [self.W_t_c]

        #initialize lists for optimizers
        self._accugrads = []  # for adadelta
        self._accugrads2 = []  # for rmsprop_graves
        self._accudeltas = []  # for adadelta

        self._accugrads.extend([common.build_shared_zeros(p.shape.eval(),
                'accugrad') for p in self.params])

        self._accugrads2.extend([common.build_shared_zeros(p.shape.eval(),
                'accugrad2') for p in self.params])

        self._accudeltas.extend([common.build_shared_zeros(p.shape.eval(),
                'accudeltas') for p in self.params])

    def _init_layers(self, params, arch_params, game_params):

        # if a trained model is given
        if params != None:
            self.W_a_0 = t.shared(params['W_a_0'], name='W_a_0', borrow=True)
            self.b_a_0 = t.shared(params['b_a_0'], name='b_a_0', borrow=True)
            self.W_a_1 = t.shared(params['W_a_1'], name='W_a_1', borrow=True)
            self.b_a_1 = t.shared(params['b_a_1'], name='b_a_1', borrow=True)
            self.W_a_2 = t.shared(params['W_a_2'], name='W_a_2', borrow=True)
            self.b_a_2 = t.shared(params['b_a_2'], name='b_a_2', borrow=True)
            self.W_a_c = t.shared(params['W_a_c'], name='W_a_c', borrow=True)

            self.W_t_0 = t.shared(params['W_t_0'], name='W_t_0', borrow=True)
            self.b_t_0 = t.shared(params['b_t_0'], name='b_t_0', borrow=True)
            self.W_t_1 = t.shared(params['W_t_1'], name='W_t_1', borrow=True)
            self.b_t_1 = t.shared(params['b_t_1'], name='b_t_1', borrow=True)
            self.W_t_2 = t.shared(params['W_t_2'], name='W_t_2', borrow=True)
            self.b_t_2 = t.shared(params['b_t_2'], name='b_t_2', borrow=True)
            self.W_t_c = t.shared(params['W_t_c'], name='W_t_c', borrow=True)

        # if no trained model is given
        else:

            # agent
            self.W_a_0 = common.create_weight(input_n=2 + 3*game_params['num_of_targets'],  output_n=arch_params['n_hidden_a_0'],   suffix='a_0')
            self.W_a_1 = common.create_weight(input_n=arch_params['n_hidden_a_0'],          output_n=arch_params['n_hidden_a_1'],   suffix='a_1')
            self.W_a_2 = common.create_weight(input_n=arch_params['n_hidden_a_1'],          output_n=arch_params['n_hidden_a_2'],   suffix='a_2')
            self.W_a_c = common.create_weight(input_n=arch_params['n_hidden_a_2'],          output_n=1,                             suffix='a_c')

            self.b_a_0 = common.create_bias(output_n=arch_params['n_hidden_a_0'], value=0, suffix='a_0')
            self.b_a_1 = common.create_bias(output_n=arch_params['n_hidden_a_1'], value=0, suffix='a_1')
            self.b_a_2 = common.create_bias(output_n=arch_params['n_hidden_a_2'], value=0, suffix='a_2')

            # transition
            self.W_t_0 = common.create_weight(input_n=2 + 3*game_params['num_of_targets']+1,    output_n=arch_params['n_hidden_t_0'],       suffix='t_0')
            self.W_t_1 = common.create_weight(input_n=arch_params['n_hidden_t_0'],              output_n=arch_params['n_hidden_t_1'],       suffix='t_1')
            self.W_t_2 = common.create_weight(input_n=arch_params['n_hidden_t_1'],              output_n=arch_params['n_hidden_t_2'],       suffix='t_2')
            self.W_t_c = common.create_weight(input_n=arch_params['n_hidden_t_2'],              output_n=3*game_params['num_of_targets'],   suffix='t_c')

            self.b_t_0 = common.create_bias(output_n=arch_params['n_hidden_t_0'], value=0, suffix='t_0')
            self.b_t_1 = common.create_bias(output_n=arch_params['n_hidden_t_1'], value=0, suffix='t_1')
            self.b_t_2 = common.create_bias(output_n=arch_params['n_hidden_t_2'], value=0, suffix='t_2')

    def _calc_params_norm(self):
        params_norm = 0
        for p in self.params:
            params_norm = params_norm + tt.sum(abs(p))

        return params_norm