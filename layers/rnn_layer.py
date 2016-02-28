import theano as t
import theano.tensor as tt
import numpy as np

class RNN_LAYER(object):

    def __init__(self, init_mean, init_var, nX, nH):

        self.Wx = t.shared(np.random.normal(init_mean, init_var, (nH, nX)).astype(t.config.floatX))
        self.Wh = t.shared(np.random.normal(init_mean, init_var, (nH, nH)).astype(t.config.floatX))
        self.b = t.shared(np.random.normal(init_mean, init_var, nH).astype(t.config.floatX))

        self.params = [self.Wx, self.Wh, self.b]

    def step(self, x, h_):

        h = tt.tanh(tt.dot(self.Wx, x) + tt.dot(self.Wh, h_) + self.b)

        return h

