import numpy as np
from svg_1 import SVG_1

import tensorflow as tf

import time
import common
import sys

class DRIVER(object):

    def __init__(self, environment, configuration, trained_model, sn_dir):

        self.env = environment

        self.config = configuration

        # with tf.device('/cpu:0')

        self.algorithm = SVG_1(configuration=self.config,
                               action_size=self.env.action_space.shape[0],
                               state_size=self.env.observation_space.shape[0])

        self.init_graph = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        if trained_model:
            self.saver.restore(self.sess, trained_model)
        else:
            self.sess.run(self.init_graph)

        self.critic_policy_switch = 1
        self.run_avg = 0.95
        self.sn_dir = sn_dir
        self.loss = 0 * np.ones(4)
        self.abs_grad = 0 * np.ones(4)
        self.abs_w = 0 * np.ones(4)
        self.time = 0
        np.set_printoptions(precision=3)
        self.batch_ = self.algorithm.ER.sample()
        self.batch_ = self.batch_ + (self.batch_[1],)
        self.gamma = 0

    def collect_experience(self, n_steps, itr, record=1, vis=0):

        observation = self.env.reset()
        t = 0
        stop = 0
        reward = 0
        
        while not stop:
            noise = np.random.normal(scale=self.config.noise_std, size=self.env.action_space.shape[0])

            if vis:
                self.env.render()
                # print 'obs: %s, reward: %f' % (observation, reward)
                # time.sleep(.5)

            if np.random.binomial(1, self.config.p_kill_noise) or vis:
                noise *= 0

            p_noise = common.multivariate_pdf_np(x=noise, mu=0., sigma=self.config.noise_std)

            if itr < self.config.model_identification_time:
                a_n = self.env.action_space.sample()
            else:
                a = self.sess.run(fetches=[self.algorithm.action],
                                  feed_dict={self.algorithm.state_: np.expand_dims(observation, 0),
                                             })

                a_n = a[0][0] + noise

            observation, reward, done, info = self.env.step(a_n)

            # debug
            # reward = self.config.reward_func_bla(observation)

            if callable(getattr(self.config, 'reward_func', None)):
                reward = self.algorithm.config.reward_func(observation)

            if record:
                self.algorithm.ER.add(action=a_n, reward=reward, state=observation, terminal=done, a_prob=p_noise)

            t += 1

            stop = done or t > n_steps

            self.time = self.run_avg * t + (1-self.run_avg) * self.time

        return

    def train_module(self, module, ind):

        batch_ = self.algorithm.ER.sample()  # states_, actions_k, rewards, states, terminals, p_k
        batch_ = batch_ + (batch_[1],)  # placeholder for actions_tm1. Doesn't go anywhere

        if ind == 2:  # when training policy: add KL constraint on policy_t w.r.t policy_t-1
            batch_ = self.batch_  # states_, actions_k, rewards, states, terminals, p_k, actions_tm1
            batch = self.algorithm.ER.sample()
            actions = self.sess.run(fetches=[self.algorithm.action],
                                    feed_dict={self.algorithm.state_: np.squeeze(batch[0])})  # states_
            batch = batch + (actions,)
            self.batch_ = batch_

        train_vals = self.sess.run(fetches=[module.minimize,
                                            module.loss,
                                            module.mean_abs_grad,
                                            module.mean_abs_w,
                                            ],
                                   feed_dict={
                                        self.algorithm.state_: np.squeeze(batch_[0]),
                                        self.algorithm.action_k: batch_[1],
                                        self.algorithm.r: batch_[2],
                                        self.algorithm.state: np.squeeze(batch_[3]),
                                        self.algorithm.p_k: batch_[5],
                                        self.algorithm.noise_std: self.config.noise_std,
                                        self.algorithm.action_tm1: batch_[6],
                                        self.algorithm.gamma: self.gamma,
                                         })

        self.loss[ind] = self.run_avg * self.loss[ind] + (1-self.run_avg) * np.asarray(train_vals[1])
        self.abs_grad[ind] = self.run_avg * self.abs_grad[ind] + (1-self.run_avg) * np.asarray(train_vals[2])
        self.abs_w[ind] = self.run_avg * self.abs_w[ind] + (1-self.run_avg) * np.asarray(train_vals[3])

    def train_step(self, itr):
        n_steps = self.config.n_steps_train
        self.collect_experience(n_steps, itr)
        self.gamma = common.calculate_gamma(self.config.intrvl, self.config.gamma_f, itr)

        for k in range(self.config.K):
            # transition
            self.train_module(self.algorithm.transition, ind=0)

            if itr > self.config.model_identification_time:
                if self.critic_policy_switch:
                    # critic
                    self.train_module(self.algorithm.critic, ind=1)
                else:
                    # policy
                    self.train_module(self.algorithm.policy, ind=2)

            # reward
            if not callable(getattr(self.config, 'reward_func', None)):
                self.train_module(self.algorithm.reward, ind=3)

            # update target weights
            if itr % self.algorithm.critic.solver_params['target_update_freq'] == 0:
                self.sess.run(self.algorithm.critic.copy_target_weights_op)

        # print progress
        if itr % 100 == 0:
            buf = self.config.train_buf_line(itr, self.loss, self.time, self.gamma)
            sys.stdout.write('\r' + buf)

        # switch critic-policy
        if itr % self.config.critic_policy_itrvl == 0:
            self.critic_policy_switch = not self.critic_policy_switch

    def test_step(self, itr):
        n_steps = self.config.n_steps_test
        self.collect_experience(n_steps, itr, vis=1, record=0)

    def print_info_line(self, itr):
        buf = self.config.test_buf_line(itr, self.loss, self.abs_grad, self.abs_w, self.time, self.gamma)
        sys.stdout.write('\r' + buf)

    def save_model(self, iter):
        common.save_params(fName=self.sn_dir + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % iter),
                           saver=self.saver, session=self.sess)
