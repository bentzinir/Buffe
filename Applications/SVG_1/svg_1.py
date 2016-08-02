import tensorflow as tf
from policy import POLICY
from transition import TRANSITION
from critic import CRITIC
from reward import REWARD
from ER import ER
import common

class SVG_1(object):

    def __init__(self, configuration, state_size, action_size, reward_size=1):

        self.config = configuration

        self.policy = POLICY(in_dim=state_size,
                             out_dim=action_size)

        self.critic = CRITIC(in_dim=state_size,
                             out_dim=1)

        self.transition = TRANSITION(in_dim=state_size +
                                            action_size,
                                     out_dim=state_size)

        # self.transition = TRANSITION_SEP(in_dim=state_size +
        #                                     action_size,
        #                              out_dim=state_size)

        self.reward = REWARD(in_dim=state_size,
                             out_dim=reward_size)

        self.ER = ER(memory_size=500000,
                     state_dim=state_size,
                     action_dim=action_size,
                     reward_dim=1,
                     batch_size=configuration.batch_size)

        # Placeholders
        # collecting experience
        self.time_vec = tf.placeholder("float")
        self.noise_std = tf.placeholder("float", shape=())

        # models optimization
        self.state_ = tf.placeholder("float")
        self.action_k = tf.placeholder("float",     shape=(self.ER.batch_size, action_size))
        self.action_tm1 = tf.placeholder("float",   shape=(self.ER.batch_size, action_size))
        self.r = tf.placeholder("float",            shape=(self.ER.batch_size, reward_size))
        self.state = tf.placeholder("float",        shape=(self.ER.batch_size, state_size))
        self.p_k = tf.placeholder("float",          shape=(self.ER.batch_size))
        self.gamma = tf.placeholder("float", shape=())

        # quick forward
        self.action = self.policy.forward(self.state_)

        w = self.importance_sampling(self.state_, self.action_k, self.p_k)

        # train transition model
        self.transition.train(state_=self.state_, action=self.action_k, state=self.state)

        # train critic
        self.critic.train(state_=self.state_, state=self.state, r=self.r, gamma=self.gamma, w=w)

        # train policy
        self.policy.train(policy_obj=self.policy_obj(state_=self.state_, w=w, a_tm1=self.action_tm1))

        # train reward
        self.reward.train(state=self.state, r=self.r)

    def policy_obj(self, state_, w, a_tm1):

        a = self.policy.forward(state_)

        state = self.transition.forward(state_, a)

        if callable(getattr(self.config, 'reward_func_tf', None)):
            r = self.config.reward_func_tf(state)
        else:
            r = self.reward.forward(state)

        R = self.critic.forward(state)

        policy_obj = tf.reduce_mean(tf.mul(w, tf.squeeze(tf.mul(self.gamma, R) + r, squeeze_dims=[1])))

        policy_obj = tf.mul(-1., policy_obj) + tf.mul(self.config.w_control, tf.nn.l2_loss(a)) + tf.mul(self.config.w_kl, tf.nn.l2_loss(a-a_tm1))

        return policy_obj

    def importance_sampling(self, state_, a_k, p_k):
        a_t = self.policy.forward(state=state_)
        p_t = common.multivariate_pdf_tf(a_k, mu=a_t, sigma=self.noise_std)
        w = tf.clip_by_value(tf.div(p_t, p_k), 0., self.policy.solver_params['max_importance_weight'])
        return w
