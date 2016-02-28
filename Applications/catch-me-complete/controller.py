import math
import theano as t
import theano.tensor as tt
import numpy as np
import common
import optimizers
from simulator import step
from gru_layer import GRU_LAYER

class CONTROLLER(object):

    def __init__(self, x_host_0, v_host_0, x_target_0, v_target_0, x_mines_0, time_steps, force, n_steps_0, n_steps_1, lr, goal_1, game_params, arch_params, solver_params, params):

        self._connect(game_params, solver_params)

        self._init_layers(params, arch_params, game_params)

        def _line_segment_point_dist(v, w, p): # x3,y3 is the point
            px = w[0] - v[0]
            py = w[1] - v[1]

            something = px*px + py*py

            u =  ((p[0] - v[0]) * px + (p[1] - v[1]) * py) / something

            # if u > 1:
            #     u = 1
            # elif u < 0:
            #     u = 0

            u = tt.nnet.relu(u) - tt.nnet.relu(u-1)

            x = v[0] + u * px
            y = v[1] + u * py

            dx = x - p[0]
            dy = y - [[1]]

            # Note: If the actual distance does not matter,
            # if you only want to compare what this function
            # returns to other results of this function, you
            # can just return the squared distance instead
            # (i.e. remove the sqrt) to gain a little performance

            dist = dx*dx + dy*dy

            return dist

        def _pdist(X,Y):
            translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
            dist = ((translation_vectors) ** 2).sum(2)
            return dist

        def _step_state(x_host_, v_host_, x_target_, v_target_, u_ext, u):

            x_target_e, v_target_e= step(x_target_, v_target_)

            # 1. approximated dynamic of the next state
            # 1.1 target (moves in fixed speed)
            v_target_a = v_target_
            x_target_a = x_target_ + self.dt * v_target_a

            # 2. difference in predictions
            # 2.1 target
            n_v_t = v_target_e - v_target_a
            n_x_t = x_target_e - x_target_a

            # 3. disconnect the gradient of the noise signals
            # 3.1 target
            n_v_t = common.disconnected_grad(n_v_t)
            n_x_t = common.disconnected_grad(n_x_t)

            # 4. add the noise to the approximation
            # 4.1 target
            v_target = v_target_a + n_v_t
            x_target = x_target_a + n_x_t

            # 5. step host
            a = (u + u_ext)*self.inv_m
            v_host_e = v_host_ + a * self.dt
            x_host_e = x_host_ + v_host_e * self.dt
            x_host, v_host = common.bounce(x_host_e, v_host_e, self.w)

            return x_host, v_host, x_target, v_target

        def _recurrence_0(time_step, f, x_host_, v_host_, x_target_, v_target_, h_0_, goal_0):
            '''
            controller 0 is responsible for navigating from current host position x_host_ to goal_0
            while neutralizing the external force

            state:
             1. dist (2): vector from current position to goal_0
             2. x (2): current position
             3. v (2): current velocity
            '''

            dist = goal_0 - x_host_

            state = tt.concatenate([dist, x_host_, v_host_])

            h_0 = self.gru_layer_0.step(state, h_0_)

            u = tt.dot(h_0, self.W_c_0) + self.b_c_0

            x_host, v_host, x_target, v_target = _step_state(x_host_, v_host_, x_target_, v_target_, f, u)

            # cost:
            discount_factor = 0.99**time_step

            # 0. smooth acceleration policy
            cost_accel = discount_factor * tt.mean(u**2)

            # 1. forcing host to move towards goal_0
            dist_from_goal = tt.mean((goal_0 - x_host)**2)

            cost_progress = discount_factor * dist_from_goal

            return (x_host, v_host, x_target, v_target, h_0, cost_accel, cost_progress)#, t.scan_module.until(dist_from_goal < 0.25)

        def _recurrence_1(x_host_, v_host_, x_target_, v_target_, h_1_, time_steps, force, x_mines, goal_1):
            '''
            controller 1 is responsible for navigating from current host position x_host_ to goal_1 while dodging mines

            state:
             1. dist (2): vector from current position to goal_1
             2. x (2): current position
             3. x_mines (2): mines locations
            '''

            dist = goal_1 - x_host_

            state = tt.concatenate([dist, x_host_, tt.flatten(x_mines)])

            h_1 = self.gru_layer_1.step(state, h_1_)

            goal_0 = tt.dot(h_1, self.W_c_1) + self.b_c_1

            goal_0 = tt.clip(goal_0, 0, self.w)

            [c_0_host_traj, c_0_host_v, c_0_target_traj, c_0_target_v, h_0, c_0_cost_accel, c_0_cost_progress], scan_updates = t.scan(fn=_recurrence_0,
                                                    sequences=[time_steps, force],
                                                    outputs_info=[x_host_, v_host_, x_target_, v_target_, self.h_0_0, None, None],
                                                    non_sequences=[goal_0],
                                                    n_steps=n_steps_0,
                                                    name='scan_func')

            x_host = c_0_host_traj[-1]
            v_host = c_0_host_v[-1]
            x_target = c_0_target_traj[-1]
            v_target = c_0_target_v[-1]

            # costs
            # 0. dodge mines
            d_host_mines = _pdist(c_0_host_traj, x_mines)
            c_1_cost_mines = tt.sum(tt.nnet.relu(self.d_mines - d_host_mines))

            # 1. forcing host to move towards goal_1
            dist_from_goal = tt.mean((goal_1 - x_host)**2)
            c_1_cost_progress = dist_from_goal

            return (x_host, v_host, x_target, v_target, h_1,
                    c_0_host_traj, c_0_target_traj, goal_0,
                    c_1_cost_mines, c_1_cost_progress, c_0_cost_accel, c_0_cost_progress)#, t.scan_module.until(dist_from_goal < 0.5)

        [c_1_host_traj, c_1_host_v, c_1_target_traj, c_1_target_v, h_1,
         c_0_host_traj, c_0_target_traj, goals_0,
         c_1_cost_mines, c_1_cost_progress, c_0_costs_accel, c_0_costs_progress], scan_updates = t.scan(fn=_recurrence_1,
                                                sequences=[],
                                                outputs_info=[x_host_0, v_host_0, x_target_0, v_target_0, self.h_1_0, None, None, None, None, None, None, None],
                                                non_sequences=[time_steps, force, x_mines_0, goal_1],
                                                n_steps=n_steps_1,
                                                name='scan_func')

        self.c_0_cost_accel = tt.mean(c_0_costs_accel)
        self.c_0_cost_progress = tt.mean(c_0_costs_progress)

        self.c_1_cost_progress = tt.mean(c_1_cost_progress)
        self.c_1_cost_mines = tt.mean(c_1_cost_mines)

        self.weighted_cost = []
        self.weighted_cost.append(
                    self.w_accel * self.c_0_cost_accel +
                    self.w_progress * self.c_0_cost_progress
        )

        self.weighted_cost.append(
                    self.w_progress * self.c_1_cost_progress +
                    self.w_mines * self.c_1_cost_mines
        )

        self.cost = []
        self.cost.append(
            self.c_0_cost_accel +
            self.c_0_cost_progress
        )

        self.cost.append(
            self.c_1_cost_progress +
            self.c_1_cost_mines
        )

        gradients = []
        gradients.append(common.create_grad_from_obj(objective=self.weighted_cost[0], params=self.param_struct[0].params, decay_val=self.l1_weight, grad_clip_val=self.grad_clip_val))
        gradients.append(common.create_grad_from_obj(objective=self.weighted_cost[1], params=self.param_struct[1].params, decay_val=self.l1_weight, grad_clip_val=self.grad_clip_val))

        updates = []
        updates.append(optimizers.optimizer(lr=lr, param_struct=self.param_struct[0], gradients=gradients[0], solver_params=solver_params))
        updates.append(optimizers.optimizer(lr=lr, param_struct=self.param_struct[1], gradients=gradients[1], solver_params=solver_params))

        # self.updates = ((updates[0]).copy()).update(updates[1])
        self.updates = updates[0]

        self.x_h = c_0_host_traj
        self.x_t = c_0_target_traj
        self.x_mines = x_mines_0
        self.goal_0 = goals_0
        self.goal_1 = goal_1

        self.grad_mean = []
        self.grad_mean.append(self._calculate_grad_mean(gradients[0]))
        self.grad_mean.append(self._calculate_grad_mean(gradients[1]))

        self.params_abs_norm = []
        self.params_abs_norm.append(self._calculate_param_abs_norm(self.param_struct[0].params))
        self.params_abs_norm.append(self._calculate_param_abs_norm(self.param_struct[1].params))

    def _connect(self, game_params, solver_params):

        self.dt = game_params['dt']
        self.w = game_params['width']
        self.inv_m = np.float32(1./game_params['m'])
        self.v_max = game_params['v_max']
        self.w_accel = game_params['w_accel']
        self.w_progress = game_params['w_progress']
        self.w_mines = game_params['w_mines']
        self.d_mines = game_params['d_mines']
        self.n_mines = game_params['n_mines']
        self.v_max = game_params['v_max']
        self.grad_clip_val = solver_params['grad_clip_val']
        self.l1_weight = solver_params['l1_weight_decay']

    def _init_layers(self, params, arch_params, game_params):

        # initialize recurrent layers value
        self.h_0_0 = np.zeros(shape=arch_params['n_hidden_0_0'], dtype=t.config.floatX)
        self.h_1_0 = np.zeros(shape=arch_params['n_hidden_1_0'], dtype=t.config.floatX)

        # if a trained model is given
        if params[0] != None:
            # c_0
            self.gru_layer_0 = GRU_LAYER(params=params[0], name='0')
            self.W_c_0 = t.shared(params['W_c_0'], name='W_c_0', borrow=True)
            self.b_c_0 = t.shared(params['b_c_0'], name='b_c_0', borrow=True)
        else:
            self.gru_layer_0 = GRU_LAYER(init_mean=0, init_var=0.1, nX=6, nH=arch_params['n_hidden_0_0'], name='gru0')

            self.W_c_0 = common.create_weight(input_n=arch_params['n_hidden_0_0'],
                                            output_n=2,
                                            suffix='0')
            self.b_c_0 = common.create_bias(output_n=2,
                                          value=0.1,
                                          suffix='0')

        if params[1] != None:
            # c_1
            self.gru_layer_1 = GRU_LAYER(params=params[1], name='1')
            self.W_c_1 = t.shared(params['W_c_1'], name='W_c_1', borrow=True)
            self.b_c_1 = t.shared(params['b_c_1'], name='b_c_1', borrow=True)
        else:
            self.gru_layer_1 = GRU_LAYER(init_mean=0, init_var=0.1, nX=4+2*self.n_mines, nH=arch_params['n_hidden_1_0'], name='gru1')

            self.W_c_1 = common.create_weight(input_n=arch_params['n_hidden_1_0'],
                                            output_n=2,
                                            suffix='1')
            self.b_c_1 = common.create_bias(output_n=2,
                                          value=0.1,
                                          suffix='1')

        self.param_struct = []
        self.param_struct.append(common.PARAMS(self.W_c_0, self.b_c_0, *self.gru_layer_0.params))
        self.param_struct.append(common.PARAMS(self.W_c_1, self.b_c_1, *self.gru_layer_1.params))

    def _calculate_param_abs_norm(self, params):
        abs_norm = 0
        for p in params:
            abs_norm += tt.mean(tt.abs_(p))
        return abs_norm

    def _calculate_grad_mean(self,gradients):
        grad_mean = 0
        for g in gradients:
            grad_mean += tt.mean(tt.abs_(g))
        return grad_mean

