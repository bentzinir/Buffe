import math
import theano as t
import theano.tensor as tt
import numpy as np
import common
import optimizers
from simulator import step

class CONTROLLER(object):

    def __init__(self,
                 x_h_0,
                 v_h_0,
                 t_h_0,
                 x_t_0,
                 v_t_0,
                 a_t_0,
                 t_t_0,
                 time_steps,
                 exist,
                 is_leader,
                 x_goal,
                 turn_vec,
                 n_steps,
                 lr,
                 game_params,
                 arch_params,
                 solver_params,
                 params):

        self._init_layers(params, arch_params, game_params)

        self._connect(game_params, solver_params)

        def _dist_from_rail(pos, rail_center, rail_radius):
            d = tt.sqrt(((pos - rail_center)**2).sum())
            return tt.sum((d - rail_radius)**2)

        def _step_state(x_h_, v_h_, angle_, speed_, t_h_, x_t_, v_t_, t_t_, ctrl, exist):

            a_t_e, v_t_e, x_t_e, t_t, t_h = step(x_h_, v_h_, t_h_, x_t_, v_t_, t_t_, exist)

            t_h = common.disconnected_grad(t_h)
            t_t = common.disconnected_grad(t_t)

            # approximated dynamic of the un-observed parts in the state
            a_t_a = tt.zeros(shape=(3,2), dtype=np.float32)

            v_t_a = v_t_

            x_t_a = x_t_ + self.dt * v_t_a

            # difference in predictions
            n_v_t = v_t_e - v_t_a

            n_a_t = a_t_e - a_t_a

            n_x_t = x_t_e - x_t_a

            # disconnect the gradient of the noise signals
            n_v_t = common.disconnected_grad(n_v_t)

            n_a_t = common.disconnected_grad(n_a_t)

            n_x_t = common.disconnected_grad(n_x_t)

            # add the noise to the approximation
            a_t = a_t_a + n_a_t

            v_t = v_t_a + n_v_t

            x_t = x_t_a + n_x_t

            # update the observed part of the state
            delta_steer = ctrl[0]
            accel = ctrl[1]

            delta_steer = tt.clip(delta_steer, -np.pi/4, np.pi/4)

            angle = angle_ + delta_steer

            speed = speed_ + accel * self.dt

            speed = tt.clip(speed, 0, self.v_max)

            v_h_x = speed * tt.sin(angle)
            v_h_y = speed * tt.cos(angle)

            v_h = tt.stack([v_h_x,v_h_y])

            x_h = x_h_ + self.dt * v_h
            x_h = tt.clip(x_h, -self.bw, self.bw)

            return x_h, v_h, angle, speed, t_h, x_t, v_t, a_t, t_t

        def _recurrence(time_step, x_h_, v_h_, angle_, speed_, t_h_, x_t_, v_t_, a_t_, t_t_, exist, is_leader, x_goal, turn_vec):
            # state
            '''
            1. host
                1.1 position (2) - (x,y) coordinates in cross coordinate system
                1.2 speed (2) - (v_x,v_y)
                # 1.3 acceleration (2) - (a_x,a_y)
                # 1.4 waiting time (1) - start counting on full stop. stop counting when clearing the junction
                1.5 x_goal (2) - destination position (indicates different turns)
                total = 5
            2. right lane car
                2.1 position (2) - null value = (-1,-1)
                2.2 speed (2) - null value = (0,0)
                2.3 acceleration (2) - null value = (0,0)
                2.4 waiting time (1) - null value = 0
                total = 7
            3. front lane car
                3.1 position (2)
                3.2 speed (2)
                3.3 acceleration (2)
                3.4 waiting time (1)
                total = 7
            4. target 3
                4.1 position (2)
                4.2 speed (2)
                4.3 acceleration (2)
                4.4 waiting time (1)
                total = 7
            total = 26
            '''

            # host_state_vec = tt.concatenate([x_h_, v_h_, t_h_])
            ang_spd = tt.stack([angle_, speed_])
            host_state_vec = tt.concatenate([x_h_, ang_spd,  x_goal])

            # target_state_vec = tt.concatenate([tt.flatten(x_t_), tt.flatten(v_t_), tt.flatten(a_t_), tt.flatten(t_t_)])
            target_state_vec = tt.concatenate([tt.flatten(x_t_), tt.flatten(v_t_), tt.flatten(a_t_), is_leader])

            state = tt.concatenate([host_state_vec, target_state_vec])

            h0 = tt.dot(state, self.W_0) + self.b_0

            relu0 = tt.nnet.relu(h0)

            h1 = tt.dot(relu0, self.W_1) + self.b_1

            relu1 = tt.nnet.relu(h1)

            h2 = tt.dot(relu1, self.W_2) + self.b_2

            relu2 = tt.nnet.relu(h2)

            a_h = tt.dot(relu2, self.W_c)

            x_h, v_h, angle, speed, t_h, x_t, v_t, a_t, t_t = _step_state(x_h_, v_h_, angle_, speed_, t_h_, x_t_, v_t_, t_t_, a_h, exist)

            # cost:

            discount_factor = 0.99**time_step

            # 0. smooth driving policy
            cost_steer = discount_factor * a_h[0]**2
            cost_accel = discount_factor * a_h[1]**2

            # 1. forcing the host to move forward
            dist_from_goal = tt.mean((x_goal - x_h)**2)

            cost_progress = discount_factor * dist_from_goal

            # 2. keeping distance from in front vehicles
            d_t_h = x_t - x_h

            h_t_dists = (d_t_h**2).sum(axis=1)

            # v_h_norm = tt.sqrt((v_h**2).sum())
            # d_t_h_norm = tt.sqrt((d_t_h**2).sum(axis=1))
            #
            # denominator = v_h_norm * d_t_h_norm
            #
            # host_targets_orientation = tt.dot(d_t_h, v_h) / (denominator + 1e-3)
            #
            # in_fornt_targets = tt.nnet.sigmoid(5 * host_targets_orientation)
            #
            # close_targets = tt.sum(tt.abs_(d_t_h))
            #
            # cost_accident = tt.mean(in_fornt_targets * close_targets)

            cost_accident = tt.sum(tt.nnet.relu(self.require_distance - h_t_dists))

            # 3. rail divergence
            cost_right_rail = _dist_from_rail(x_h, self.right_rail_center, self.right_rail_radius) * turn_vec[0]
            cost_front_rail = (x_h[0] - self.lw/2)**2 * turn_vec[1]
            cost_left_rail = _dist_from_rail(x_h, self.left_rail_center, self.left_rail_radius) * turn_vec[2]

            cost_rail = cost_right_rail + cost_left_rail + cost_front_rail

            return (x_h, v_h, angle, speed, t_h, x_t, v_t, a_t, t_t,
                    cost_steer, cost_accel, cost_progress, cost_accident, cost_rail, a_h), t.scan_module.until(dist_from_goal < 0.001)

        [x_h, v_h, angle, speed, t_h, x_t, v_t, a_t, t_t,
                costs_steer, costs_accel, costs_progress, costs_accident, costs_rail, a_hs], scan_updates = t.scan(fn=_recurrence,
                                                sequences=time_steps,
                                                outputs_info=[x_h_0, v_h_0, 0., 0., t_h_0, x_t_0, v_t_0, a_t_0, t_t_0,
                                                              None, None, None, None, None, None],
                                                non_sequences=[exist, is_leader, x_goal, turn_vec],
                                                n_steps=n_steps,
                                                name='scan_func')

        # 3. right of way cost term

        T = x_h.shape[0]

        x_h_rpt_1 = tt.repeat(x_h,T,axis=1) # (Tx2T)

        x_h_rpt_1_3d = x_h_rpt_1.dimshuffle(0,1,'x') # (Tx2Tx1)

        x_h_3D = tt.repeat(x_h_rpt_1_3d, 3, axis=2) # (Tx2Tx3)

        x_t_rshp_1 = tt.zeros(shape=(2*T,3),dtype=np.float32) # (2Tx3)

        x_t_rshp_1_x = tt.set_subtensor(x_t_rshp_1[:T,:],x_t[:,:,0])

        x_t_rshp_1_xy = tt.set_subtensor(x_t_rshp_1_x[T:,:],x_t[:,:,1])

        x_t_rshp_1_3d = x_t_rshp_1_xy.dimshuffle(0,1,'x') # (2Tx3x1)

        x_t_rpt_2_3d = tt.repeat(x_t_rshp_1_3d,T,axis=2) # (2Tx3xT)

        x_t_3D = x_t_rpt_2_3d.dimshuffle(2,0,1) # (Tx2Tx3)

        # abs_diff_mat = tt.abs_(x_h_3D - x_t_3D) # (Tx2Tx3)
        abs_diff_mat = (x_h_3D - x_t_3D)**2 # (Tx2Tx3)

        dists_mat = abs_diff_mat[:,:T,:] + abs_diff_mat[:,T:,:] # d_x+d_y: (TxTx3)

        # punish only when cutting a leader
        host_effective_dists = (tt.triu(dists_mat[:,:,0]) * is_leader[0] +
                                tt.triu(dists_mat[:,:,1]) * is_leader[1] +
                                tt.triu(dists_mat[:,:,2]) * is_leader[2])

        costs_row = tt.mean(tt.nnet.sigmoid(self.eps_row - host_effective_dists))

        self.cost_steer = tt.mean(costs_steer)
        self.cost_accel = tt.mean(costs_accel)
        self.cost_progress = tt.mean(costs_progress)
        self.cost_accident = tt.mean(costs_accident)
        self.cost_row = tt.mean(costs_row)
        self.cost_rail = tt.mean(costs_rail)

        self.weighted_cost = (
                    self.w_delta_steer * self.cost_steer +
                    self.w_accel * self.cost_accel +
                    self.w_progress * self.cost_progress +
                    self.w_accident * self.cost_accident +
                    # self.w_row * self.cost_row
                    self.w_rail * self.cost_rail
        )

        self.cost = (
            self.cost_steer +
            self.cost_accel +
            self.cost_progress +
            self.cost_accident +
            # self.cost_row
            self.cost_rail
        )

        objective = self.weighted_cost

        objective = common.weight_decay(objective=objective, params=self.params, l1_weight=self.l1_weight)

        objective = t.gradient.grad_clip(objective, -self.grad_clip_val, self.grad_clip_val)

        gradients = tt.grad(objective, self.params)

        self.updates = optimizers.optimizer(lr=lr, param_struct=self, gradients=gradients, solver_params=solver_params)

        self.x_h = x_h
        self.v_h = v_h
        self.x_t = x_t
        self.v_t = v_t

        self.max_a = tt.max(abs(a_hs))

        self.max_grad_val = 0
        self.grad_mean = 0
        for g in gradients:
            self.grad_mean += tt.mean(tt.abs_(g))
            self.max_grad_val = (tt.max(g) > self.max_grad_val) * tt.max(g) + (tt.max(g) <= self.max_grad_val) * self.max_grad_val

        self.params_abs_norm = self._calc_params_norm()

    def _connect(self, game_params, solver_params):

        self.dt = game_params['dt']
        self.w_delta_steer = game_params['w_delta_steer']
        self.w_accel = game_params['w_accel']
        self.w_progress = game_params['w_progress']
        self.w_accident = game_params['w_accident']
        self.w_row = game_params['w_row']
        self.w_rail = game_params['w_rail']
        self.require_distance = game_params['require_distance']
        self.v0 = game_params['v0']
        self.v_max = game_params['v_max']
        self.lw =game_params['lane_width']
        self.bw = game_params['board_width']
        self.host_length = game_params['host_length']
        self.eps_row = game_params['eps_row']
        self.right_rail_center = np.asarray([self.lw, -self.lw], dtype=np.float32)
        self.left_rail_center = np.asarray([-self.lw, -self.lw], dtype=np.float32)
        self.right_rail_radius = self.lw/2
        self.left_rail_radius = 3*self.lw/2
        self.grad_clip_val = solver_params['grad_clip_val']

        self.l1_weight = solver_params['l1_weight_decay']
        self.params = []
        self.params += [self.W_0, self.b_0]
        self.params += [self.W_1, self.b_1]
        self.params += [self.W_2, self.b_2]
        self.params += [self.W_c]

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
            # load gru_layer params
            # self.gru_layer = GRU_LAYER(params=params)
            self.W_0 = t.shared(params['W_0'], name='W_0', borrow=True)
            self.b_0 = t.shared(params['b_0'], name='b_0', borrow=True)
            self.W_1 = t.shared(params['W_1'], name='W_1', borrow=True)
            self.b_1 = t.shared(params['b_1'], name='b_1', borrow=True)
            self.W_2 = t.shared(params['W_2'], name='W_2', borrow=True)
            self.b_2 = t.shared(params['b_2'], name='b_2', borrow=True)
            self.W_c = t.shared(params['W_c'], name='W_c', borrow=True)

        # if no trained model is given
        else:
            self.W_0 = common.create_weight(input_n=27,
                                            output_n=arch_params['n_hidden_0'],
                                            suffix='0')
            self.b_0 = common.create_bias(output_n=arch_params['n_hidden_0'],
                                          value=0.1,
                                          suffix='0')

            self.W_1 = common.create_weight(input_n=arch_params['n_hidden_0'],
                                            output_n=arch_params['n_hidden_1'],
                                            suffix='1')
            self.b_1 = common.create_bias(output_n=arch_params['n_hidden_1'],
                                          value=0,
                                          suffix='1')

            self.W_2 = common.create_weight(input_n=arch_params['n_hidden_1'],
                                            output_n=arch_params['n_hidden_2'],
                                            suffix='2')
            self.b_2 = common.create_bias(output_n=arch_params['n_hidden_2'],
                                          value=0,
                                          suffix='2')

            # initialize classifier Weight
            self.W_c = common.create_weight(input_n=arch_params['n_hidden_2'],
                                            output_n=2,
                                            suffix='c')

    def _calc_params_norm(self):
        params_norm = 0
        for p in self.params:
            params_norm = params_norm + tt.sum(abs(p))

        return params_norm