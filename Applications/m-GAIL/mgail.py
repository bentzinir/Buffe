from policy import POLICY
from transition import TRANSITION
from discriminator import DISCRIMINATOR
from ER import ER
import numpy as np
import tensorflow as tf
import common

class MGAIL(object):

    def __init__(self, environment, configuration, state_size, action_size, scan_size, expert_data=None):

        self.env = environment

        self.policy = POLICY(in_dim=state_size,
                             out_dim=1)

        self.transition = TRANSITION(in_dim=state_size+action_size,
                                     out_dim=state_size)

        self.discriminator = DISCRIMINATOR(in_dim=state_size+action_size,
                                           out_dim=2)

        self.er_agent = ER(memory_size=200000,
                           state_dim=state_size,
                           reward_dim=1,
                           batch_size=configuration.batch_size)

        if expert_data is None:
            self.er_expert = ER(memory_size=200000,
                                state_dim=state_size,
                                reward_dim=1,
                                batch_size=configuration.batch_size)

        else:
            self.er_expert = common.load_er(fname=expert_data,
                                            batch_size=configuration.batch_size,
                                            history_length=1,
                                            state_dim=state_size)

        # policy placeholders
        self.time_vec = tf.placeholder("float", shape=(None,))
        self.scan_0 = tf.placeholder("float", shape=scan_size)
        self.gamma = tf.placeholder("float", shape=())

        # transition / discriminator placeholders
        self.state_ = tf.placeholder("float", shape=(None, state_size))
        self.action = tf.placeholder("float", shape=(None,))
        self.state = tf.placeholder("float", shape=(None, state_size))
        self.labels = tf.placeholder("float", shape=(None, 1))
        self.states = tf.placeholder("float", shape=(None, state_size))
        self.actions = tf.placeholder("float", shape=(None,))

        # train transition model
        self.transition.train(state_=self.state_,
                              action=self.action,
                              state=self.state)

        # train discriminator
        self.discriminator.train(states=self.states,
                                 actions=self.actions,
                                 labels=self.labels,
                                 )
        # calculate discriminator accuracy
        self.discriminator.zero_one_loss(states=self.states,
                                 actions=self.actions,
                                 labels=self.labels,
                                         )

        def _recurrence(scan_, time):

            state_ = tf.slice(scan_, [0], [self.env.state_size])

            if expert_data is None:
                a = self.env.expert(state_)
            else:
                a = self.policy.forward(state_)

            a = tf.squeeze(a, squeeze_dims=[1])

            d = self.discriminator.forward(tf.expand_dims(state_, 0), a)

            # minimize the gap between p_agent and p_expert
            # d[:,0] = p_agent
            # d[:,1] = p_expert
            p_agent = tf.squeeze(tf.slice(d, [0, 0], [-1, 1]), squeeze_dims=[1])
            p_expert = tf.squeeze(tf.slice(d, [0, 1], [-1, 1]), squeeze_dims=[1])
            p_gap = p_agent - p_expert

            # DEBUG here !!!
            step_cost = tf.mul(tf.pow(self.gamma, time), p_gap)

            # get next state
            state_e = self.env.step(scan_, a)

            state_a = self.transition.forward(tf.expand_dims(state_, 0), a)

            state = self.re_parametrization(state_e=state_e, state_a=tf.squeeze(state_a, squeeze_dims=[0]))

            return self.env.pack_scan(state, a, step_cost)

        self.scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.scan_0)

        self.policy_loss, self.policy_loss_weighted = self.env.total_loss(self.scan_returns)

        # train the policy
        self.policy.train(policy_obj=self.policy_loss_weighted)

    def push_er(self, module, trajectory):
        actions, _, states, N = self.env.slice_scan(trajectory)
        rewards = np.zeros_like(actions)
        terminals = np.zeros_like(actions)
        terminals[-1] = 1
        module.add(actions=actions, rewards=rewards, states=states, terminals=terminals)

    def re_parametrization(self, state_e, state_a):
        n = state_e - state_a
        n = tf.stop_gradient(n)
        return state_a + n

