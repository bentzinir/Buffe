from policy import POLICY
from transition import TRANSITION
from discriminator import DISCRIMINATOR
from ER import ER
import numpy as np
import tensorflow as tf

class MGAIL(object):

    def __init__(self, environment, state_size, action_size, scan_size, expert_data=None):

        self.env = environment

        self.policy = POLICY(in_dim=state_size,
                             out_dim=1)

        self.transition = TRANSITION(in_dim=state_size+action_size,
                                     out_dim=state_size)

        self.discriminator = DISCRIMINATOR(in_dim=state_size+action_size,
                                           out_dim=1)

        self.er_agent = ER(memory_size=200000,
                           state_dim=state_size,
                           reward_dim=3,
                           batch_size=32)

        self.er_expert = ER(memory_size=200000,
                            state_dim=state_size,
                            reward_dim=3,
                            batch_size=32)

        # DEBUG : work with random expert data
        self.er_expert.count = 199999
        # self.load_expert_experience(er=self.er_expert,
        #                             expert_data=expert_data,
        #                             environment=self.env)

        # policy placeholders
        self.time_vec = tf.placeholder("float")
        self.scan_0 = tf.placeholder("float", scan_size)

        # transition / discriminator placeholders
        self.state_ = tf.placeholder("float")
        self.action = tf.placeholder("float")
        self.state = tf.placeholder("float")
        self.state_e = tf.placeholder("float")
        self.action_e = tf.placeholder("float")

        # train transition model
        self.transition.train(state_=self.state_,
                              action=self.action,
                              state=self.state)

        # train discriminator
        self.discriminator.train(state=self.state,
                                 action=self.action,
                                 state_e=self.state_e,
                                 action_e=self.action_e
                                 )

        def _recurrence(scan_, time):

            state_ = tf.slice(scan_, [0], [self.env.state_size])

            a = self.policy.forward(tf.expand_dims(state_, 0))

            state_e = self.env.step(scan_, a)

            state_a = self.transition.forward(tf.expand_dims(state_, 0), a)

            state = self.re_parametrization(state_e=state_e, state_a=tf.squeeze(state_a, squeeze_dims=[0]))

            # step_costs = self.simulator.step_loss(state, a, time)

            step_cost = self.discriminator.forward(state_=state_a, action=a)

            return self.env.pack_scan(state, a, scan_, step_cost)

        self.scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.scan_0)

        self.policy_loss, self.policy_loss_weighted = self.env.total_loss(self.scan_returns)

        # train the policy
        self.policy.train(policy_obj=self.policy_loss_weighted)

    def push_er_agent(self, returned_scan_vals):
        actions, _, states, N = self.env.slice_scan(returned_scan_vals)
        rewards = np.zeros_like(actions)
        terminals = np.zeros_like(actions)
        terminals[-1] = 1
        self.er_agent.add(actions=actions, rewards=rewards, states=states, terminals=terminals)

    def re_parametrization(self, state_e, state_a):
        n = state_e - state_a
        n = tf.stop_gradient(n)
        return state_a + n

    def scalar_to_4D(self, x):
        return tf.expand_dims(tf.expand_dims(tf.expand_dims(x, -1), -1), -1)
