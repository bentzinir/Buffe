from ER import ER
import tensorflow as tf
import common

class MGAIL(object):

    def __init__(self, environment):

        self.env = environment

        self.do_keep_prob = tf.placeholder("float", shape=(), name='do_keep_prob')

        self.sparse_ae = __import__('sparse_ae').SPARSE_AE(in_dim=self.env.state_size,
                                                           hidden_dim=self.env.sae_hidden_size)

        if self.env.use_sae:
            autoencoder = self.sparse_ae
            transformed_state_size = self.env.sae_hidden_size
        else:
            autoencoder = None
            transformed_state_size = self.env.state_size

        self.transition = __import__('transition').TRANSITION(in_dim=transformed_state_size+self.env.action_size,
                                                              out_dim=self.env.state_size,
                                                              size=self.env.t_size,
                                                              lr=self.env.t_lr,
                                                              do_keep_prob=self.do_keep_prob)

        self.discriminator = __import__('discriminator').DISCRIMINATOR(in_dim=transformed_state_size + self.env.action_size,
                                                                       out_dim=1+1*self.env.disc_as_classifier,
                                                                       size=self.env.d_size,
                                                                       lr=self.env.d_lr,
                                                                       do_keep_prob=self.do_keep_prob)

        self.policy = __import__('policy').POLICY(in_dim=transformed_state_size,
                                                  out_dim=self.env.action_size,
                                                  size=self.env.p_size,
                                                  lr=self.env.p_lr,
                                                  w_std=self.env.w_std,
                                                  do_keep_prob=self.do_keep_prob,
                                                  n_accum_steps=self.env.policy_accum_steps)

        self.policy_ = __import__('policy').POLICY(in_dim=transformed_state_size,
                                                   out_dim=self.env.action_size,
                                                   size=self.env.p_size,
                                                   lr=self.env.p_lr,
                                                   w_std=self.env.w_std,
                                                   do_keep_prob=self.do_keep_prob,
                                                   n_accum_steps=self.env.policy_accum_steps)

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

        # TODO: add prioritized sweeping buffer
        # TODO: punish on h0 variance

        self.env.sigma = self.er_expert.actions_std/self.env.noise_intensity
        self.states = tf.placeholder("float", shape=(None, None, self.env.state_size), name='states')  # Time x Batch x State
        self.actions = tf.placeholder("float", shape=(None, None, self.env.action_size), name='action')  # Time x Batch x Action
        self.label = tf.placeholder("float", shape=(None, 1), name='label')
        self.gamma = tf.placeholder("float", shape=(), name='gamma')

        states = common.normalize(self.states, self.er_expert.states_mean, self.er_expert.states_std)
        actions = common.normalize(self.actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        state = tf.squeeze(states, squeeze_dims=[0])  # 1 x Batch x State ==> Batch x State
        action = tf.squeeze(actions, squeeze_dims=[0])  # 1 x Batch x Action ==> Batch x Action

        # 0. Sparse Autoencoder
        self.h0, self.h1 = self.sparse_ae.forward(state)
        rho_hat = tf.reduce_mean(self.h0, 0)
        rho_rho_hat_kl = tf.reduce_sum(common.kl_div(self.env.sae_rho, rho_hat))
        sparse_ae_loss = tf.nn.l2_loss(self.h1 - state)/float(self.env.sae_batch) + self.env.sae_beta * rho_rho_hat_kl
        self.sparse_ae.train(objective=sparse_ae_loss)

        # 1. Transition
        def transition_loop(state_, action):
            state_a = self.transition.forward(state_, action, autoencoder)
            return state_a

        states_0 = tf.squeeze(tf.slice(states, [0, 0, 0], [1, -1, -1]), squeeze_dims=[0])
        states_1_to_T = tf.slice(states, [1, 0, 0], [-1, -1, -1])
        states_1_to_T_a = tf.scan(transition_loop, elems=actions, initializer=states_0)
        transition_loss = tf.nn.l2_loss(states_1_to_T - states_1_to_T_a)
        self.transition.train(objective=transition_loss)

        # measure accuracy only on 1-step prediction
        states_1 = tf.slice(states_1_to_T, [0, 0, 0], [1, -1, -1])
        states_1_a = tf.slice(states_1_to_T_a, [0, 0, 0], [1, -1, -1])
        self.transition.acc = tf.nn.l2_loss(states_1 - states_1_a)

        # 2. Discriminator
        labels = tf.concat(1, [1 - self.label, self.label])
        d = self.discriminator.forward(state, action, autoencoder)

        if self.env.disc_as_classifier:  # treat as a classifier
            # 2.1 0-1 accuracy
            predictions = tf.argmax(input=d, dimension=1)
            correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
            self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            # 2.2 prediction
            discriminator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels))
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
        self.action_test = common.denormalize(self.policy.forward(state, autoencoder), self.er_expert.actions_mean, self.er_expert.actions_std)

        # 4. Policy
        # 4.1 SL
        action_a = self.policy.forward(state, autoencoder)
        policy_sl_loss = tf.reduce_mean(tf.abs(action_a - action))  # action == expert action
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
            a = self.policy.forward(state_, autoencoder)
            eta = self.env.sigma * tf.random_normal(shape=tf.shape(a))
            a += eta

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            d = self.discriminator.forward(state_, a, autoencoder)
            cost = self.al_loss(d)

            # discount the cost
            step_cost = tf.mul(tf.pow(self.gamma, t), cost)
            total_cost += step_cost

            # get next state
            a_denormed = common.denormalize(a, self.er_expert.actions_mean, self.er_expert.actions_std)
            state_env, reward, env_term_sig, info = self.env.step(a_denormed, mode='tensorflow')
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std)
            state_e = tf.stop_gradient(state_e)
            state_a = self.transition.forward(state_, a, autoencoder)
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
        return tf.nn.l2_loss(tf.mul(logit_gap, self.env.policy_al_w * tf.to_float(logit_gap > 0)))
