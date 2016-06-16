from scipy.stats import norm
import tensorflow as tf
from policy import POLICY
from transition import TRANSITION
from critic import CRITIC
from ER import ER
import numpy as np

class SVG_1(object):

    def __init__(self, simulator):

        self.simulator = simulator

        self.policy = POLICY(in_dim=self.simulator.state_size,
                             out_dim=1)

        self.critic = CRITIC(in_dim=self.simulator.state_size,
                             out_dim=1)

        self.transition = TRANSITION(in_dim=self.simulator.state_size+1,
                                     out_dim=self.simulator.state_size)

        self.ER = ER(memory_size=200000,
                     state_dim=self.simulator.state_size,
                     reward_dim=3,
                     batch_size=32)

        self.lr = tf.placeholder("float")

        # Placeholders
        # collecting experience
        self.time_vec = tf.placeholder("float")
        self.scan_0 = tf.placeholder("float", shape=self.simulator.scan_size)
        self.noise_std = tf.placeholder("float")

        # models optimization
        self.state_ = tf.placeholder("float", shape=(self.ER.batch_size, self.simulator.state_size))
        self.action = tf.placeholder("float", shape=(self.ER.batch_size))
        self.state_e = tf.placeholder("float", shape=(self.ER.batch_size, self.simulator.state_size))
        self.p_k = tf.placeholder("float", shape=(self.ER.batch_size))

        # interact with env to collect experience
        self.scan_returns = self.collect_experience()

        # train transition model
        self.minimize_t, self.loss_t = self.train_transition()

        # train critic
        self.minimize_c, self.loss_c = self.train_critic()

        # train policy
        self.minimize_p, self.loss_p = self.train_policy()

    def policy_obj(self, state_, w):

        a = self.policy.forward(state_)

        l = self.simulator.step_loss(state_, a, time=0.)

        l = self.simulator.weight_costs(l)

        state = self.transition.forward(state_, tf.expand_dims(a, 1))

        L = self.critic.forward(state)

        policy_obj = tf.reduce_mean(tf.mul(w, tf.reduce_sum(tf.mul(self.simulator.gamma, L) + l)))

        return policy_obj

    def collect_experience(self):

        def _recurrence(scan_, time):
            state_ = tf.slice(scan_, [0], [self.simulator.state_size])

            a = self.policy.forward(tf.expand_dims(state_, 0))

            noise = tf.random_normal(shape=a.get_shape(), stddev=self.noise_std)

            a += noise

            state = self.simulator.step(scan_, a)

            step_costs = self.simulator.step_loss(tf.expand_dims(state, 0), a, time)

            return self.simulator.pack_scan(state, scan_, a, noise, step_costs)

        scan_returns = tf.scan(fn=_recurrence, elems=self.time_vec, initializer=self.scan_0)

        return scan_returns

    def push_ER(self, returned_scan_vals):
        actions, noise, rewards, states, N = self.simulator.slice_scan(returned_scan_vals)
        terminals = np.zeros_like(actions)
        terminals[-1] = 1
        a_probs = norm.pdf(noise, loc=0, scale=self.ER.n_std)

        self.ER.add(actions=actions, rewards=rewards, states=states, terminals=terminals, a_probs=a_probs)

    def train_transition(self):
        state_a = self.transition.forward(self.state_, tf.expand_dims(self.action,1))
        loss = self.transition.loss(state_a, self.state_e)
        minimize = self.transition.backward(loss)
        return minimize, loss

    def train_critic(self):
        v_state = self.critic.forward(self.state_)
        v_state_t = self.critic.forward_t(self.state_)
        loss = self.critic.loss(v_state, v_state_t) # TODO : change to real loss after implementing target network
        minimize = self.critic.backward(loss)
        return minimize, loss

    def train_policy(self):
        w = self.importance_sampling(self.state_, self.action, self.p_k)
        loss = self.policy_obj(state_=self.state_, w=w)
        minimize = self.policy.backward(loss)
        return minimize, loss

    def importance_sampling(self, state_, a_k, p_k):
        a_t = self.policy.forward(state=state_)
        p_t = self.normal_pdf(a_k, mu=a_t, sigma=self.noise_std)
        w = tf.clip_by_value(tf.div(p_t, p_k), 0., self.policy.solver_params['max_importance_weight'])
        return w

    def normal_pdf(self, x, mu, sigma):
        p_x = tf.mul(tf.div(1., tf.mul(np.sqrt(2*np.pi).astype(np.float32), sigma)), tf.exp(tf.div(tf.square(x-mu), tf.mul(-2., tf.square(sigma)))))
        p_x = tf.stop_gradient(p_x)
        return p_x
