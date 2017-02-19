from ER import ER
import tensorflow as tf
import common
import numpy as np


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
                                                                      lstm=self.env.fm_lstm,
                                                                      dropout_keep=self.env.do_keep_prob)
        autoencoder = None
        transformed_state_size = self.env.state_size

        self.discriminator = __import__('discriminator').DISCRIMINATOR(in_dim=transformed_state_size + self.env.action_size,
                                                                       out_dim=2,
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

        # self.policy_ = __import__('policy').POLICY(in_dim=transformed_state_size,
        #                                            out_dim=self.env.action_size,
        #                                            size=self.env.p_size,
        #                                            lr=self.env.p_lr,
        #                                            w_std=self.env.w_std,
        #                                            do_keep_prob=self.do_keep_prob,
        #                                            n_accum_steps=self.env.policy_accum_steps,
        #                                            weight_decay=self.env.weight_decay)

        self.er_agent = ER(memory_size=self.env.er_agent_size,
                           state_dim=self.env.state_size,
                           action_dim=self.env.action_size,
                           reward_dim=1,  # stub connection
                           qpos_dim=self.env.qpos_size,
                           qvel_dim=self.env.qvel_size,
                           batch_size=self.env.batch_size,
                           history_length=1)

        self.er_expert = common.load_er(fname=self.env.run_dir + self.env.expert_data,
                                        batch_size=self.env.batch_size,
                                        history_length=1,
                                        traj_length=2)

        self.env.sigma = self.er_expert.actions_std/self.env.noise_intensity
        self.states_ = tf.placeholder("float", shape=(None, self.env.state_size), name='states_')  # Batch x State
        self.states = tf.placeholder("float", shape=(None, self.env.state_size), name='states')  # Batch x State
        self.actions = tf.placeholder("float", shape=(None, self.env.action_size), name='action')  # Batch x Action
        self.label = tf.placeholder("float", shape=(None, 1), name='label')
        self.gamma = tf.placeholder("float", shape=(), name='gamma')
        self.temp = tf.placeholder("float", shape=(), name='temperature')
        self.noise = tf.placeholder("float", shape=(), name='noise_flag')
        self.noise_mean = tf.placeholder("float", shape=(self.env.action_size))

        states_ = common.normalize(self.states_, self.er_expert.states_mean, self.er_expert.states_std)
        states = common.normalize(self.states, self.er_expert.states_mean, self.er_expert.states_std)
        if self.env.continuous_actions:
            actions = common.normalize(self.actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        else:
            actions = self.actions

        self.forward_model.states_normalizer = self.er_expert.states_max - self.er_expert.states_min
        self.forward_model.actions_normalizer = self.er_expert.actions_max - self.er_expert.actions_min
        self.forward_model.states_normalizer = self.er_expert.states_std
        self.forward_model.actions_normalizer = self.er_expert.actions_std
        s = np.ones((1, self.forward_model.arch_params['encoding_dim']))

        # 1. Forward Model
        fm_output, _, gru_state = self.forward_model.forward([states_, actions, s])
        l2_loss = tf.reduce_mean(tf.square(states-fm_output))
        self.forward_model.train(objective=l2_loss)

        # 2. Discriminator
        labels = tf.concat(1, [1 - self.label, self.label])
        d = self.discriminator.forward(states, actions, autoencoder)

        # 2.1 0-1 accuracy
        correct_predictions = tf.equal(tf.argmax(d, 1), tf.argmax(labels, 1))
        self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        # 2.2 prediction
        d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
        # cost sensitive weighting (weigh true=exprt, predict=agent mistakes)
        d_loss_weighted = self.env.cost_sensitive_weight * tf.mul(tf.to_float(tf.equal(tf.squeeze(self.label), 1.)), d_cross_entropy) +\
                                                           tf.mul(tf.to_float(tf.equal(tf.squeeze(self.label), 0.)), d_cross_entropy)
        discriminator_loss = tf.reduce_mean(d_loss_weighted)

        self.discriminator.train(objective=discriminator_loss)
        self.discriminator.acc_summary = tf.scalar_summary('acc_d', self.discriminator.acc)

        # 3. Collect experience
        mu = self.policy.forward(states, autoencoder)
        if self.env.continuous_actions:
            a = common.denormalize(mu, self.er_expert.actions_mean, self.er_expert.actions_std)
            eta = tf.random_normal(shape=tf.shape(a), stddev=self.env.sigma, mean=self.noise_mean)
            self.action_test = tf.squeeze(a + self.noise * eta)
        else:
            a = common.gumbel_softmax(logits=mu, temperature=self.temp)
            self.action_test = tf.argmax(a, dimension=1)

        # 4. Policy
        # 4.1 SL
        actions_a = self.policy.forward(states, autoencoder)
        policy_sl_loss = tf.nn.l2_loss(actions_a - actions)  # action == expert action
        self.policy.train(objective=policy_sl_loss, mode='sl')

        # 4.2 Temporal Regularization
        actions_a_ = self.policy_.forward(states, autoencoder)
        policy_tr_loss = self.env.policy_tr_w * self.env.policy_accum_steps * tf.nn.l2_loss(actions_a - actions_a_)
        self.policy.train(objective=policy_tr_loss, mode='tr')
        # op for copying weights from policy to policy_
        self.policy_.copy_weights(self.policy.weights, self.policy.biases)

        # Plain adversarial learning
        d = self.discriminator.forward(states, actions_a, autoencoder)
        policy_alr_loss = self.al_loss(d)
        self.policy.train(objective=policy_alr_loss, mode='alr')

        # 4.3 AL
        def policy_loop(state_, t, total_cost, total_trans_err, _):
            mu = self.policy.forward(state_, autoencoder)

            if self.env.continuous_actions:
                eta = self.env.sigma * tf.random_normal(shape=tf.shape(mu), mean=self.noise_mean)
                a = mu + eta
            else:
                a = common.gumbel_softmax_sample(logits=mu, temperature=self.temp)

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            d = self.discriminator.forward(state_, a, autoencoder)
            cost = self.al_loss(d)

            # add step cost
            total_cost += tf.mul(tf.pow(self.gamma, t), cost)

            # get next state
            if self.env.continuous_actions:
                a_sim = common.denormalize(a, self.er_expert.actions_mean, self.er_expert.actions_std)
            else:
                a_sim = tf.argmax(a, dimension=1)

            state_env, _, env_term_sig, = self.env.step(a_sim, mode='tensorflow')[:3]
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std)
            state_e = tf.stop_gradient(state_e)

            state_a, _, _ = self.forward_model.forward([state_, a, s])

            state, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
            total_trans_err += tf.reduce_mean(abs(nu))
            t += 1

            return state, t, total_cost, total_trans_err, env_term_sig

        def policy_stop_condition(state_, t, cost, trans_err, env_term_sig):
            cond = tf.logical_not(env_term_sig)
            cond = tf.logical_and(cond, t < self.env.n_steps_train)
            cond = tf.logical_and(cond, trans_err < self.env.total_trans_err_allowed)
            return cond

        state_0 = tf.slice(states, [0, 0], [1, -1])
        loop_outputs = tf.while_loop(policy_stop_condition, policy_loop, [state_0, 0., 0., 0., False])
        self.policy.train(objective=loop_outputs[2], mode='al')


    def al_loss(self, d):
        logit_agent, logit_expert = tf.split(split_dim=1, num_split=2, value=d)
        logit_gap = logit_agent - logit_expert
        valid_cond = tf.stop_gradient(tf.to_float(logit_gap > 0))
        valid_gaps = tf.mul(logit_gap, valid_cond)

        # L2
        if self.env.al_loss == 'L2':
            loss = tf.nn.l2_loss(tf.mul(logit_gap, tf.to_float(logit_gap > 0)))
        # L1
        elif self.env.al_loss == 'L1':
            loss = tf.reduce_mean(valid_gaps)
        # Cross entropy
        elif self.env.al_loss == 'CE':
            labels = tf.concat(1, [tf.zeros_like(logit_agent), tf.ones_like(logit_expert)])
            d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
            loss = tf.reduce_mean(d_cross_entropy)

        return loss*self.env.policy_al_w
