import tensorflow as tf
from policy import POLICY
from transition import TRANSITION
from transition_sep import TRANSITION_SEP
from ER import ER
import numpy as np

class SVG_INF(object):

    def __init__(self, simulator):

        self.simulator = simulator

        self.policy = POLICY(in_dim=self.simulator.state_size, out_dim=1)

        self.transition = TRANSITION(in_dim=self.simulator.state_size+1,
                                     out_dim=self.simulator.state_size)

        self.ER = ER(memory_size=200000,
                     state_dim=self.simulator.state_size,
                     reward_dim=3,
                     batch_size=32)

        self.lr = tf.placeholder("float")

        # for policy
        self.time_vec = tf.placeholder("float")
        self.scan_0 = tf.placeholder("float", shape=self.simulator.scan_size)

        # for transition
        self.state_ = tf.placeholder("float")
        self.action = tf.placeholder("float")
        self.state_e = tf.placeholder("float")

        def _recurrence(scan_, time):

            state_ = tf.slice(scan_, [0], [self.simulator.state_size])

            state_ = tf.expand_dims(state_, 0)

            a = self.policy.forward(state_)

            state_e = self.simulator.step(scan_, a)

            state_a = self.transition.forward(state_, tf.expand_dims(a, 1))

            state = self.re_parametrization(state_e=state_e, state_a=tf.squeeze(state_a, squeeze_dims=[0]))

            step_costs = self.simulator.step_loss(state, a, time)

            return self.simulator.pack_scan(state, scan_, a, step_costs)

        self.scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.scan_0)

        self.policy_loss, self.policy_loss_weighted = self.simulator.total_loss(self.scan_returns)

        # train the policy
        self.minimize_controller = self.policy.backward(loss=self.policy_loss_weighted)

        # train the transition model
        self.train_transition()

    def push_ER(self, returned_scan_vals):
        actions, rewards, states, N = self.simulator.slice_scan(returned_scan_vals)
        terminals = np.zeros_like(actions)
        terminals[-1] = 1
        self.ER.add(actions=actions, rewards=rewards, states=states, terminals=terminals)

    def train_transition(self):
        state_a = self.transition.forward(tf.squeeze(self.state_), tf.expand_dims(self.action, 1))
        self.transition_loss = self.transition.loss(state_a, tf.squeeze(self.state_e))
        self.minimize_transition = self.transition.backward(self.transition_loss)

    def re_parametrization(self, state_e, state_a):
        n = state_e - state_a
        tf.stop_gradient(n)
        return state_a + n

    def scalar_to_4D(self, x):
        return tf.expand_dims(tf.expand_dims(tf.expand_dims(x, -1), -1), -1)

