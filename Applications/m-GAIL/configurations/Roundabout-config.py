import tensorflow as tf
import datetime

class CONFIGURATION(object):

    def __init__(self):

        self.n_train_iters = 1000000
        self.test_interval = 1000
        self.n_steps_train = 200
        self.n_steps_test = 1000

        self.action_size = 1
        self.model_identification_time = 10000
        self.p_kill_noise = 0.
        self.noise_std = 0.7
        self.K = 2
        self.phase = 20e3
        self.gamma_f = 0.9
        self.intrvl = 500
        self.gamma = 0
        self.batch_size = 48
        self.alpha_accel = 0.5,
        self.alpha_progress = 0.5,
        self.alpha_accident = 0.5,
        self.w_kl = 1e-4

    def train_buf_line(self, itr, loss, time_, gamma):
        buf = "processing iter: %d, loss(transition,discriminator,policy): %s, average time: %f, gamma: %f" % (
            itr, loss, time_, gamma)
        return buf

    def test_buf_line(self, itr, loss, abs_grad, abs_w, time_, gamma):
        buf = '%s Training: iter %d, loss: %s, grads: %s, weights: %s, average time: %f, gamma: %f\n' % \
            (datetime.datetime.now(), itr, loss, abs_grad, abs_w, time_, gamma)
        return buf
