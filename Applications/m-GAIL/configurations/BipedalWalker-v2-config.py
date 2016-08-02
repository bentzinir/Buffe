import tensorflow as tf
import datetime

class CONFIGURATION(object):

    def __init__(self):

        self.n_train_iters = 1000000
        self.test_interval = 1000
        self.n_steps_train = 200
        self.n_steps_test = 1000

        self.action_size = 1
        self.critic_policy_itrvl = 1000
        self.model_identification_time = 10000
        self.p_kill_noise = 0.
        self.noise_std = 0.7
        self.K = 2
        self.phase = 20e3
        self.gamma_f = 0.9
        self.intrvl = 500
        self.gamma = 0
        self.batch_size = 48
        self.w_pole_angle = 2e-1
        self.w_angular_val = 1e-4
        self.w_cart_position = 1e-4
        self.w_control = 5e-4  # 1e-4
        self.w_kl = 1e-4

    def reward_func(self, state):
        cart_position = state[0]
        pole_angle = state[1]
        cart_vel = state[2]
        pole_angular_vel = state[3]

        cost = (self.w_pole_angle * abs(
            pole_angle) + self.w_angular_val * pole_angular_vel ** 2 + self.w_cart_position * cart_position)
        return -cost

    def reward_func_tf(self, state):
        cart_position = tf.slice(state, [0, 0], [-1, 1])
        pole_angle = tf.slice(state, [0, 1], [-1, 1])
        cart_vel = tf.slice(state, [0, 2], [-1, 1])
        pole_angular_vel = tf.slice(state, [0, 3], [-1, 1])

        cost = tf.mul(self.w_pole_angle, tf.abs(pole_angle)) + tf.mul(self.w_angular_val,
                                                                      tf.square(pole_angular_vel)) + tf.mul(
            self.w_cart_position, cart_position)
        return -cost

    def train_buf_line(self, itr, loss, time_, gamma):
        buf = "processing iter: %d, loss(transition,critic,policy,reward): %s, average time: %f, gamma: %f" % (
            itr, loss, time_, gamma)
        return buf

    def test_buf_line(self, itr, loss, abs_grad, abs_w, time_, gamma):
        buf = '%s Training ctrlr 1: iter %d, loss: %s, grads: %s, weights: %s, average time: %f, gamma: %f\n' % \
            (datetime.datetime.now(), itr, loss, abs_grad, abs_w, time_, gamma)
        return buf
