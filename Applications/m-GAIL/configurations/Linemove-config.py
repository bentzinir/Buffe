import datetime

class CONFIGURATION(object):

    def __init__(self):

        self.name = 'Roundabout'
        self.n_train_iters = 1000000
        self.test_interval = 2000
        self.n_steps_train = 100
        self.n_steps_test = 100
        self.n_episodes_expert = 50
        self.action_size = 1
        self.model_identification_time = 0
        self.discr_policy_itrvl = 200
        self.K = 1
        self.gamma = 0.99
        self.batch_size = 3
        self.alpha_accel = 0.5,
        self.alpha_progress = 0.5,
        self.alpha_accident = 0.5,
        self.w_kl = 1e-4

    def info_line(self, itr, loss, discriminator_acc, abs_grad=None, abs_w=None):
        if abs_grad is not None:
            buf = '%s Training: iter %d, loss: %s, disc_acc: %f, grads: %s, weights: %s\n' % \
                (datetime.datetime.now(), itr, loss, discriminator_acc, abs_grad, abs_w)
        else:
            buf = "processing iter: %d, loss(transition,discriminator,policy): %s, disc_acc: %f" % (itr, loss, discriminator_acc)
        return buf
