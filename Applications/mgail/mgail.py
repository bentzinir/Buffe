from ER import ER
import tensorflow as tf
import common


class MGAIL(object):

    def __init__(self, environment):

        self.env = environment

        self.do_keep_prob = tf.placeholder("float", shape=(), name='do_keep_prob')

        self.forward_model = __import__('forward_model').ForwardModel(state_size=self.env.state_size,
                                                                      action_size=self.env.action_size,
                                                                      rho=self.env.fm_rho,
                                                                      beta=self.env.fm_beta,
                                                                      encoding_size=self.env.fm_encoding_size,
                                                                      batch_size=self.env.fm_batch_size,
                                                                      multi_layered_encoder=self.env.fm_multi_layered_encoder,
                                                                      num_steps=self.env.fm_num_steps,
                                                                      separate_encoders=self.env.fm_separate_encoders,
                                                                      merger=self.env.fm_merger,
                                                                      activation=self.env.fm_activation,
                                                                      lstm=self.env.fm_lstm)
        autoencoder = None
        transformed_state_size = self.env.state_size

        self.discriminator = __import__('discriminator').DISCRIMINATOR(in_dim=transformed_state_size + self.env.action_size,
                                                                       out_dim=1+1*self.env.disc_as_classifier,
                                                                       size=self.env.d_size,
                                                                       lr=self.env.d_lr,
                                                                       do_keep_prob=self.do_keep_prob,
                                                                       weight_decay=self.env.weight_decay)

        self.policy = __import__('policy').POLICY(in_dim=transformed_state_size,
                                                  out_dim=self.env.action_size,
                                                  size=self.env.p_size,
                                                  lr=self.env.p_lr,
                                                  w_std=self.env.w_std,
                                                  do_keep_prob=self.do_keep_prob,
                                                  n_accum_steps=self.env.policy_accum_steps,
                                                  weight_decay=self.env.weight_decay)

        self.policy_ = __import__('policy').POLICY(in_dim=transformed_state_size,
                                                   out_dim=self.env.action_size,
                                                   size=self.env.p_size,
                                                   lr=self.env.p_lr,
                                                   w_std=self.env.w_std,
                                                   do_keep_prob=self.do_keep_prob,
                                                   n_accum_steps=self.env.policy_accum_steps,
                                                   weight_decay=self.env.weight_decay)

        self.er_agent = ER(memory_size=self.env.er_agent_size,
                           state_dim=self.env.state_size,
                           action_dim=self.env.action_size,
                           reward_dim=1,  # stub connection
                           batch_size=self.env.batch_size,
                           history_length=1)

        self.er_expert = common.load_er(fname=self.env.run_dir + self.env.expert_data,
                                        batch_size=self.env.batch_size,
                                        history_length=1,
                                        traj_length=2)

        self.env.sigma = self.er_expert.actions_std/self.env.noise_intensity
        self.states = tf.placeholder("float", shape=(None, None, self.env.state_size), name='states')  # Time x Batch x State
        self.actions = tf.placeholder("float", shape=(None, None, self.env.action_size), name='action')  # Time x Batch x Action
        self.label = tf.placeholder("float", shape=(None, 1), name='label')
        self.gamma = tf.placeholder("float", shape=(), name='gamma')
        self.temp = tf.placeholder("float", shape=(), name='temperature')
        self.noise = tf.placeholder("float", shape=(), name='noise_flag')

        states = common.normalize(self.states, self.er_expert.states_mean, self.er_expert.states_std)
        if self.env.continuous_actions:
            actions = common.normalize(self.actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        else:
            actions = self.actions

        state = tf.squeeze(states, squeeze_dims=[0])  # 1 x Batch x State ==> Batch x State
        action = tf.squeeze(actions, squeeze_dims=[0])  # 1 x Batch x Action ==> Batch x Action

        # 0. Pretrain Forward Model
        print('Pretraining the forward model for ' + str(self.env.fm_num_iterations) + ' iterations')
        self.forward_model.pretrain(self.env.fm_opt, self.env.fm_lr, self.env.fm_batch_size, self.env.fm_num_iterations, self.env.fm_expert_er_path)

        # 1. Forward Model
        def forward_model_loop(state_, action):
            state_ /= self.forward_model.states_normalizer
            action /= self.forward_model.actions_normalizer
            input = [state_, action]
            state_a, _ = self.forward_model.forward(input)
            state_a *= self.forward_model.states_normalizer
            state_ *= self.forward_model.states_normalizer
            action *= self.forward_model.actions_normalizer

            return state_a

        states_0 = tf.squeeze(tf.slice(states, [0, 0, 0], [1, -1, -1]), squeeze_dims=[0])
        states_1_to_T = tf.slice(states, [1, 0, 0], [-1, -1, -1])
        states_1_to_T_a = tf.scan(forward_model_loop, elems=actions, initializer=states_0)
        forward_model_loss = tf.nn.l2_loss(states_1_to_T - states_1_to_T_a)
        self.forward_model.train(objective=forward_model_loss)

        # measure accuracy only on 1-step prediction
        states_1 = tf.slice(states_1_to_T, [0, 0, 0], [1, -1, -1])
        states_1_a = tf.slice(states_1_to_T_a, [0, 0, 0], [1, -1, -1])
        self.forward_model.acc = tf.nn.l2_loss(states_1 - states_1_a)

        # 2. Discriminator
        labels = tf.concat(1, [1 - self.label, self.label])
        d = self.discriminator.forward(state, action, autoencoder)

        if self.env.disc_as_classifier:  # treat as a classifier
            # 2.1 0-1 accuracy
            correct_predictions = tf.equal(tf.argmax(d, 1), tf.argmax(labels, 1))
            self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            # 2.2 prediction
            d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
            # cost sensitive weighting (weigh true=exprt, predict=agent mistakes)
            d_loss_weighted = self.env.cost_sensitive_weight * tf.mul(tf.to_float(tf.equal(tf.squeeze(self.label), 1.)), d_cross_entropy) +\
                                                               tf.mul(tf.to_float(tf.equal(tf.squeeze(self.label), 0.)), d_cross_entropy)
            discriminator_loss = tf.reduce_mean(d_loss_weighted)

        else:  # treat as a regressor
            # 2.1 0-1 accuracy
            self.discriminator.acc = tf.reduce_mean(tf.to_float(tf.logical_or(
                tf.logical_and(tf.greater_equal(d, 0.5), tf.equal(self.label, 1)),
                tf.logical_and(tf.less_equal(d, 0.5), tf.equal(self.label, 0)))))
            # 2.2 prediction
            discriminator_loss = tf.nn.l2_loss(d-self.label)

        self.discriminator.train(objective=discriminator_loss)
        self.discriminator.acc_summary = tf.scalar_summary('acc_d', self.discriminator.acc)

        # 3. Collect experience
        mu = self.policy.forward(state, autoencoder)
        if self.env.continuous_actions:
            a = common.denormalize(mu, self.er_expert.actions_mean, self.er_expert.actions_std)
            eta = tf.random_normal(shape=tf.shape(a), stddev=self.env.sigma)
            self.action_test = tf.squeeze(a + self.noise * eta)
        else:
            a = common.gumbel_softmax(logits=mu, temperature=self.temp)
            self.action_test = tf.argmax(a, dimension=1)
            pass

        # 4. Policy
        # 4.1 SL
        action_a = self.policy.forward(state, autoencoder)
        policy_sl_loss = tf.reduce_mean(tf.abs(action_a - action))  # action == expert action
        # TODO: checking euclidean loss
        policy_sl_loss = tf.nn.l2_loss(1 * (action_a - action) )  # action == expert action
        self.policy.train(objective=policy_sl_loss, mode='sl')
        self.policy.loss_sl_summary = tf.scalar_summary('loss_p_sl', self.policy.loss_sl)

        # 4.2 Temporal Regularization
        action_a_ = self.policy_.forward(state, autoencoder)
        policy_tr_loss = self.env.policy_tr_w * self.env.policy_accum_steps * tf.nn.l2_loss(action_a - action_a_)
        self.policy.train(objective=policy_tr_loss, mode='tr')
        self.policy.loss_tr_summary = tf.scalar_summary('loss_p_tr', self.policy.loss_tr)
        # op for copying weights from policy to policy_
        self.policy_.copy_weights(self.policy.weights, self.policy.biases)

        # Plain adversarial learning
        d = self.discriminator.forward(state, action_a, autoencoder)
        policy_alr_loss = self.al_loss(d)
        self.policy.train(objective=policy_alr_loss, mode='alr')

        # 4.3 AL
        def policy_loop(state_, t, total_cost, total_trans_err, _):
            mu = self.policy.forward(state_, autoencoder)

            if self.env.continuous_actions:
                # supposed to be the correct way
                # eta = tf.random_normal(shape=tf.shape(mu), stddev=1./self.env.noise_intensity)

                # works better
                eta = self.env.sigma * tf.random_normal(shape=tf.shape(mu))
                a = mu + eta
            else:
                # a_sample = common.choice(mu)
                # eta = a_sample - mu
                # eta = tf.stop_gradient(eta)
                # a = mu + eta
                a = common.gumbel_softmax_sample(logits=mu, temperature=self.temp)

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            d = self.discriminator.forward(state_, a, autoencoder)
            cost = self.al_loss(d)

            # discount the cost
            step_cost = tf.mul(tf.pow(self.gamma, t), cost)
            total_cost += step_cost

            # get next state
            if self.env.continuous_actions:
                a_sim = common.denormalize(a, self.er_expert.actions_mean, self.er_expert.actions_std)
            else:
                a_sim = tf.argmax(a, dimension=1)

            state_env, reward, env_term_sig, info = self.env.step(a_sim, mode='tensorflow')
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std)
            state_e = tf.stop_gradient(state_e)

            state_ /= self.forward_model.states_normalizer
            a /= self.forward_model.actions_normalizer
            input = [state_, a]
            state_a, _ = self.forward_model.forward(input)
            state_a *= self.forward_model.states_normalizer
            state_ *= self.forward_model.states_normalizer

            state, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
            total_trans_err += tf.reduce_mean(abs(nu))
            t += 1

            return state, t, total_cost, total_trans_err, env_term_sig

        def policy_stop_condition(state_, t, cost, trans_err, env_term_sig):
            cond = tf.logical_not(env_term_sig)
            cond = tf.logical_and(cond, t < self.env.n_steps_train)
            cond = tf.logical_and(cond, trans_err < self.env.total_trans_err_allowed)
            return cond

        state_0 = tf.slice(state, [0, 0], [1, -1])
        loop_outputs = tf.while_loop(policy_stop_condition, policy_loop, [state_0, 0., 0., 0., False], parallel_iterations=1, back_prop=True)
        policy_al_loss = self.env.policy_al_w * loop_outputs[2]
        self.policy.train(objective=policy_al_loss, mode='al')

        self.policy.loop_time = loop_outputs[1]

        self.policy.loss_al_summary = tf.scalar_summary('loss_p_al', self.policy.loss_al)

    def al_loss(self, d):
        logit_agent, logit_expert = tf.split(split_dim=1, num_split=2, value=d)
        logit_gap = logit_agent - logit_expert
        loss = tf.nn.l2_loss(tf.mul(logit_gap, self.env.policy_al_w * tf.to_float(logit_gap > 0)))
        return loss
