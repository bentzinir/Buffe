from mgail import MGAIL
import numpy as np
import tensorflow as tf
import time
import common
import sys


class DRIVER(object):

    def __init__(self, environment):

        self.env = environment
        self.algorithm = MGAIL(environment=self.env)
        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.test_writer = tf.train.SummaryWriter(self.env.run_dir + '/test')
        self.sess = tf.Session()
        if self.env.trained_model:
            self.saver.restore(self.sess, self.env.trained_model)
        else:
            self.sess.run(self.init_graph)
        self.run_dir = self.env.run_dir
        self.loss = 999. * np.ones(3)
        self.abs_grad = 0 * np.ones(3)
        self.abs_w = 0 * np.ones(3)
        self.reward = 0
        self.reward_std = 0
        self.run_avg = 0.001
        self.discriminator_policy_switch = 0
        self.policy_loop_time = 0
        self.disc_acc = 0
        self.er_count = 0
        self.itr = 0
        self.best_reward = 0
        self.mode = 'SL'
        np.set_printoptions(precision=2)
        np.set_printoptions(linewidth=220)

    def module_to_index(self, module):
        v = {'transition': 0, 'discriminator': 1, 'policy': 2}
        return v[module]

    def update_stats(self, module, attr, value):
        module_ind = self.module_to_index(module)
        if attr == 'loss':
            self.loss[module_ind] = self.run_avg * self.loss[module_ind] + (1-self.run_avg) * np.asarray(value)
        elif attr == 'grad':
            self.abs_grad[module_ind] = self.run_avg * self.abs_grad[module_ind] + (1 - self.run_avg) * np.asarray(value)
        elif attr == 'weights':
            self.abs_w[module_ind] = self.run_avg * self.abs_w[module_ind] + (1 - self.run_avg) * np.asarray(value)
        elif attr == 'accuracy':
            self.disc_acc = self.run_avg * self.disc_acc + (1 - self.run_avg) * np.asarray(value)

    def train_autoencoder(self):
        alg = self.algorithm
        if self.itr < self.env.model_identification_time:
            er = self.algorithm.er_expert
        else:
            er = self.algorithm.er_agent
        states = []
        for i in range(int(self.env.sae_batch/self.env.batch_size)):
            state_, action, _, _, _ = er.sample()
            state_ = np.squeeze(np.copy(state_), axis=1)
            states.append(state_)
        states = np.reshape(np.array(states), (1, -1, self.env.state_size))
        fetches = [alg.sparse_ae.minimize, alg.sparse_ae.loss,
                   alg.sparse_ae.mean_abs_grad, alg.sparse_ae.mean_abs_w]
        feed_dict = {alg.states: states}
        self.run_tensorflow(fetches, feed_dict, ind=3)

    def train_transition(self):
        alg = self.algorithm
        for k_t in range(self.env.K_T):
            # trans_iters = self.num_transition_iters()
            # state_, action = self.algorithm.er_agent.sample_trajectory(trans_iters)
            # states = np.transpose(state_, axes=[1, 0, 2])
            # actions = np.transpose(action, axes=[1, 0, 2])
            #if np.random.randint(0, 2) == 1:
            #    states_, actions, _, states, terminals = self.algorithm.er_expert.sample()
            #else:
            states_, actions, _, states, terminals = self.algorithm.er_agent.sample()
            states_ = np.squeeze(states_, axis=1)
            states = np.squeeze(states, axis=1)
            fetches = [alg.forward_model.minimize, alg.forward_model.loss,
                       alg.forward_model.mean_abs_grad, alg.forward_model.mean_abs_w]
            feed_dict = {alg.states_: states_, alg.states: states, alg.actions: actions, alg.do_keep_prob: self.env.do_keep_prob}
            run_vals = self.sess.run(fetches, feed_dict)
            self.update_stats('transition', 'loss', run_vals[1])
            self.update_stats('transition', 'grad', run_vals[2])
            self.update_stats('transition', 'weights', run_vals[3])

    def train_discriminator(self):
        alg = self.algorithm
        for k_d in range(self.env.K_D):
            state_a_, action_a, _, _, _ = self.algorithm.er_agent.sample()
            state_e_, action_e, _, _, _ = self.algorithm.er_expert.sample()
            state_a_ = np.squeeze(state_a_, axis=1)
            state_e_ = np.squeeze(state_e_, axis=1)
            states = np.concatenate([state_a_, state_e_])
            actions = np.concatenate([action_a, action_e])

            # labels (policy/expert) : 0/1, and in 1-hot form: policy-[1,0], expert-[0,1]
            labels_a = np.zeros(shape=(alg.er_agent.batch_size,))
            labels_e = np.ones(shape=(alg.er_agent.batch_size,))
            labels = np.expand_dims(np.concatenate([labels_a, labels_e]), axis=1)
            fetches = [alg.discriminator.minimize, alg.discriminator.loss, alg.discriminator.mean_abs_grad,
                       alg.discriminator.mean_abs_w, alg.discriminator.acc]
            feed_dict = {alg.states: states, alg.actions: actions,
                         alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
            run_vals = self.sess.run(fetches, feed_dict)
            self.update_stats('discriminator', 'loss', run_vals[1])
            self.update_stats('discriminator', 'grad', run_vals[2])
            self.update_stats('discriminator', 'weights', run_vals[3])
            self.update_stats('discriminator', 'accuracy', run_vals[4])

    def train_policy(self, mode):
        alg = self.algorithm
        for k_p in range(self.env.K_P):

            # reset the policy gradient
            self.sess.run([alg.policy.reset_grad_op], {})

            state_e_, action_e, _, _, _ = self.algorithm.er_expert.sample()
            state_e_ = np.squeeze(state_e_, axis=1)

            if mode == 'SL':
                # accumulate the SL gradient
                fetches = [alg.policy.accum_grads_sl, alg.policy.loss_sl]
                feed_dict = {alg.states: state_e_, alg.actions: action_e, alg.do_keep_prob: self.env.do_keep_prob}
                run_vals = self.sess.run(fetches, feed_dict)
                self.update_stats('policy', 'loss', run_vals[1])

                # apply SL gradient
                self.sess.run([alg.policy.apply_grads_sl], {})

                # output gradient / weights statistics
                run_vals = self.sess.run([alg.policy.mean_abs_grad_sl, alg.policy.mean_abs_w_sl], {})
                self.update_stats('policy', 'grad', run_vals[0])
                self.update_stats('policy', 'weights', run_vals[1])

                # copy weights: w_policy_ <- w_policy
                self.sess.run([alg.policy_.copy_weights_op], {})

            else:  # Adversarial Learning
                if self.env.get_status():
                    state = self.env.reset()
                    self.episode_noise_shift = np.random.normal(scale=alg.env.sigma)
                else:
                    state = self.env.get_state()

                # Accumulate the (noisy) adversarial gradient
                for i in range(self.env.policy_accum_steps):
                    # accumulate AL gradient
                    fetches = [alg.policy.accum_grads_al, alg.policy.loss_al]
                    feed_dict = {alg.states: np.array([state]), alg.gamma: self.env.gamma,
                                 alg.do_keep_prob: self.env.do_keep_prob, alg.noise: 1., alg.temp: self.env.temp,
                                 alg.noise_mean: self.episode_noise_shift}
                    run_vals = self.sess.run(fetches, feed_dict)
                    self.update_stats('policy', 'loss', run_vals[1])

                # Plain Adversarial Learning
                # fetches = [alg.policy.accum_grads_alr, alg.policy.loss_alr]
                # feed_dict = {alg.states: np.array(state_e_), alg.do_keep_prob: 1.}
                # run_vals = self.sess.run(fetches, feed_dict)
                # self.update_stats('policy', 'loss', run_vals[1])

                # Temporal Regularization
                # TODO: disabled dropout for tr loss
                self.sess.run([alg.policy.accum_grads_tr], {alg.states: state_e_, alg.do_keep_prob: 1.})

                # copy weights: w_policy_ <- w_policy
                self.sess.run([alg.policy_.copy_weights_op], {})

                # apply AL gradient
                self.sess.run([alg.policy.apply_grads_al], {})

                # output gradient / weights statistics
                run_vals = self.sess.run([alg.policy.mean_abs_grad_al, alg.policy.mean_abs_w_al], {})
                self.update_stats('policy', 'grad', run_vals[0])
                self.update_stats('policy', 'weights', run_vals[1])

    def collect_experience(self, record=1, vis=0, n_steps=None, noise_flag=True):
        alg = self.algorithm
        observation = self.env.reset()
        self.episode_noise_shift = np.random.normal(scale=alg.env.sigma)
        do_keep_prob = self.env.do_keep_prob
        t = 0
        R = 0
        done = 0
        if n_steps is None:
            n_steps = self.env.n_steps_test

        while not done:
            if vis:
                self.env.render()

            if not noise_flag:
                do_keep_prob = 1.

            a = self.sess.run(fetches=[alg.action_test], feed_dict={alg.states: np.reshape(observation, [1, -1]),
                                                                    alg.do_keep_prob: do_keep_prob,
                                                                    alg.noise: noise_flag,
                                                                    alg.noise_mean: self.episode_noise_shift,
                                                                    alg.temp: self.env.temp})

            observation, reward, done, info = self.env.step(a, mode='python')

            done = done or t > n_steps

            t += 1

            R += reward

            if record:
                if self.env.continuous_actions:
                    action = a
                else:
                    action = np.zeros((1, self.env.action_size))
                    action[0, a[0]] = 1
                self.algorithm.er_agent.add(actions=action, rewards=[reward], next_states=[observation], terminals=[done])

        return R

    def reset_module(self, module):

        temp = set(tf.all_variables())

        module.backward(module.loss)

        self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    def train_step(self):
        # phase_0: Model identification:
        # autoencoder: learning from expert data
        # forward_model: learning from the expert data
        # discriminator: learning concurrently with policy
        # policy: learning in SL mode

        # phase_1: Adversarial training
        # autoencoder: learning from agent data
        # forward_model: learning from agent data
        # discriminator: learning in an interleaved mode with policy
        # policy: learning in adversarial mode

        # Train policy in SL
        if self.itr < self.env.model_identification_time:
            self.mode = 'SL'
            if self.env.train_flags[2]:
                self.train_policy(self.mode)

        # Fill Experience Buffer
        elif self.itr == self.env.model_identification_time and self.env.pre_load_buffer:
            while self.algorithm.er_agent.current == self.algorithm.er_agent.count:
                self.collect_experience()
                buf = 'Collecting examples...%d/%d' % (self.algorithm.er_agent.current, self.algorithm.er_agent.states.shape[0])
                sys.stdout.write('\r' + buf)

        # Adversarial Learning
        else:
            if self.env.train_flags[1]:
                self.train_transition()

            if self.env.train_flags[2]:
                self.mode = 'Prep'
                if self.itr < (self.env.model_identification_time + self.env.prep_time):
                    self.train_discriminator()
                else:
                    self.mode = 'AL'
                    if self.discriminator_policy_switch:
                        self.train_discriminator()
                    else:
                        self.train_policy(self.mode)

                        if self.itr % self.env.collect_experience_interval == 0:
                            self.collect_experience()

                    # switch discriminator-policy
                    if self.itr % self.env.discr_policy_itrvl == 0:
                        self.discriminator_policy_switch = not self.discriminator_policy_switch

        # print progress
        if self.itr % 100 == 0:
            self.print_info_line('slim')

    # def num_transition_iters(self):
    #     if self.loss[0] > self.env.trans_loss_th:
    #         trans_iters = 2
    #     else:
    #         trans_iters = int((2.-self.env.max_trans_iters)/self.env.trans_loss_th * self.loss[0] + self.env.max_trans_iters)
    #     return trans_iters

    def print_info_line(self, mode):
        if mode == 'full':
            buf = '%s Training(%s): iter %d, loss: %s, disc_acc: %f, grads: %s, weights: %s, er_count: %d, R: %.1f, R_std: %.2f\n' % \
                  (time.strftime("%H:%M:%S"), self.mode, self.itr, self.loss, self.disc_acc, self.abs_grad, self.abs_w,
                   self.algorithm.er_agent.count, self.reward, self.reward_std)
            if hasattr(self.env, 'log_fid'):
                self.env.log_fid.write(buf)
        else:
            buf = "processing iter: %d, loss(transition,discriminator,policy): %s, disc_acc: %f" % (self.itr, self.loss, self.disc_acc)
        sys.stdout.write('\r' + buf)

    def save_model(self, dir_name=None):
        if dir_name is None:
            dir_name = self.run_dir + '/snapshots/'
        fname= dir_name + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % self.itr)
        common.save_params(fname=fname, saver=self.saver, session=self.sess)
