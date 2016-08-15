from ER import ER
import numpy as np
import tensorflow as tf
import common


class MGAIL(object):

    def __init__(self, environment):

        TRANSITION = __import__('transition').TRANSITION

        DISCRIMINATOR = __import__('discriminator').DISCRIMINATOR

        POLICY = __import__('policy').POLICY

        self.env = environment

        self.policy = POLICY(in_dim=self.env.state_size,
                             out_dim=self.env.action_size)

        self.transition = TRANSITION(in_dim=self.env.state_size+self.env.action_size,
                                     out_dim=self.env.state_size)

        self.discriminator = DISCRIMINATOR(in_dim=self.env.state_size+self.env.action_size,
                                           out_dim=2)

        self.er_agent = ER(memory_size=self.env.er_agent_size,
                           state_dim=self.env.state_size,
                           action_dim=self.env.action_size,
                           reward_dim=1,  # stub connection
                           batch_size=self.env.batch_size)

        if self.env.expert_data is None:
            self.er_expert = ER(memory_size=200000,
                                state_dim=self.env.state_size,
                                action_dim=self.env.action_size,
                                reward_dim=1,
                                batch_size=self.env.batch_size)

        else:
            self.er_expert = common.load_er(fname=self.env.run_dir + 'expert-' + self.env.expert_data,
                                            batch_size=self.env.batch_size,
                                            history_length=1,
                                            state_dim=self.env.state_size,
                                            action_dim=self.env.action_size)

        # policy placeholders
        self.time_vec = tf.placeholder("float", shape=(None,))
        self.scan_0 = tf.placeholder("float", shape=(self.env.scan_batch, self.env.scan_size))
        self.gamma = tf.placeholder("float", shape=())
        self.sigma = tf.placeholder("float", shape=())

        # transition / discriminator placeholders
        self.states_ = tf.placeholder("float", shape=(None, self.env.state_size))
        self.states = tf.placeholder("float", shape=(None, self.env.state_size))
        self.actions = tf.placeholder("float", shape=(None, self.env.action_size))
        self.labels = tf.placeholder("float", shape=(None, 1))

        # train transition model
        self.transition.train(state_=self.states_,
                              action=self.actions,
                              state=self.states)

        # train discriminator
        self.discriminator.train(states=self.states_,
                                 actions=self.actions,
                                 labels=self.labels,
                                 )

        # calculate discriminator accuracy
        self.discriminator.zero_one_loss(states=self.states_,
                                         actions=self.actions,
                                         labels=self.labels,
                                         )

        def _recurrence(scan_, time):

            state_ = tf.slice(scan_, [0, 0], [-1, self.env.state_size])

            a = self.policy.forward(state_)

            eta = tf.mul(self.sigma, tf.random_normal(shape=tf.shape(a)))

            a += eta

            d = self.discriminator.forward(state_, a)

            # minimize the gap between p_agent (d[:,0]) and p_expert (d[:,1])
            p_agent = tf.slice(d, [0, 0], [-1, 1])
            p_expert = tf.slice(d, [0, 1], [-1, 1])
            p_gap = p_agent - p_expert

            # clip p_gap for stability
            p_gap = tf.clip_by_value(p_gap, -self.env.max_p_gap, self.env.max_p_gap)

            step_cost = tf.mul(tf.pow(self.gamma, time), p_gap)

            # get next state
            state_e = self.env.step(scan_, a)

            state_a = self.transition.forward(state_, a)

            state = self.re_parametrization(state_e=state_e, state_a=state_a)

            return self.env.pack_scan(state, a, step_cost, scan_)

        self.scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.scan_0)

        self.policy_loss, self.policy_loss_weighted = self.env.total_loss(self.scan_returns)

        # train the policy
        self.policy.train(policy_obj=self.policy_loss_weighted)

    def push_er(self, module, trajectory):
        for i in range(self.env.scan_batch):
            trajectory_i = trajectory[:, i, :]
            actions, _, states, N = self.env.slice_scan(trajectory_i)
            rewards = np.zeros(N)
            terminals = np.zeros(N)
            terminals[-1] = 1
            module.add(actions=actions, rewards=rewards, states=states, terminals=terminals)

    def re_parametrization(self, state_e, state_a):
        n = state_e - state_a
        n = tf.stop_gradient(n)
        return state_a + n
