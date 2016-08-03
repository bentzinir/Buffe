import datetime

class CONFIGURATION(object):

    def __init__(self):

        self.name = 'Roundabout'
        self.n_train_iters = 1000000
        self.test_interval = 10000
        self.n_steps_train = 50
        self.n_steps_test = 50
        self.n_episodes_expert = 50
        self.action_size = 1
        self.model_identification_time = 0
        self.discr_policy_itrvl = 100000
        self.K = 1
        self.gamma = 0.9
        self.batch_size = 48
        self.alpha_accel = 0.5,
        self.alpha_progress = 0.5,
        self.alpha_accident = 0.5,
        self.w_kl = 1e-4


    def train_buf_line(self, itr, loss):
        buf = "processing iter: %d, loss(transition,discriminator,policy): %s" % (itr, loss)
        return buf

    def test_buf_line(self, itr, loss, abs_grad, abs_w):
        buf = '%s Training: iter %d, loss: %s, grads: %s, weights: %s\n' % \
            (datetime.datetime.now(), itr, loss, abs_grad, abs_w)
        return buf
