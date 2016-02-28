import theano as t
import theano.tensor as tt
import numpy as np
from collections import OrderedDict

class LSTM_LAYER(object):

    def __init__(self, init_mean, init_var, nX, nH):

        # self.Wxi = t.shared(np.random.normal(init_mean, init_var, (nH, nX)).astype(t.config.floatX), name='Wxi')
        # self.Wxj = t.shared(np.random.normal(init_mean, init_var, (nH, nX)).astype(t.config.floatX), name='Wxj')
        # self.Wxf = t.shared(np.random.normal(init_mean, init_var, (nH, nX)).astype(t.config.floatX), name='Wxf')
        # self.Wxo = t.shared(np.random.normal(init_mean, init_var, (nH, nX)).astype(t.config.floatX), name='Wxo')
        # self.Whi = t.shared(np.random.normal(init_mean, init_var, (nH, nH)).astype(t.config.floatX), name='Whi')
        # self.Whj = t.shared(np.random.normal(init_mean, init_var, (nH, nH)).astype(t.config.floatX), name='Whj')
        # self.Whf = t.shared(np.random.normal(init_mean, init_var, (nH, nH)).astype(t.config.floatX), name='Whf')
        # self.Who = t.shared(np.random.normal(init_mean, init_var, (nH, nH)).astype(t.config.floatX), name='Who')

        self.Wi = t.shared(np.random.normal(init_mean, init_var, (nH, nX+nH)).astype(t.config.floatX), name='Wi')
        self.Wj = t.shared(np.random.normal(init_mean, init_var, (nH, nX+nH)).astype(t.config.floatX), name='Wj')
        self.Wx = t.shared(np.random.normal(init_mean, init_var, (nH, nX+nH)).astype(t.config.floatX), name='Wf')
        self.Wo = t.shared(np.random.normal(init_mean, init_var, (nH, nX+nH)).astype(t.config.floatX), name='Wo')

        self.bi = t.shared(np.random.normal(init_mean, init_var, nH).astype(t.config.floatX), name='bi')
        self.bj = t.shared(np.random.normal(init_mean, init_var, nH).astype(t.config.floatX), name='bj')
        self.bf = t.shared(np.ones(nH, dtype=t.config.floatX), name='bf')
        self.bo = t.shared(np.random.normal(init_mean, init_var, nH).astype(t.config.floatX), name='bo')

        # self.params = [self.Wxi, self.Wxj, self.Wxf, self.Wxo,\
        #                self.Whi, self.Whj, self.Whf, self.Who, \
        #                self.bi, self.bj, self.bf, self.bo]

        self.params = [self.Wi, self.Wj, self.Wf, self.Wo,
                       self.bi, self.bj, self.bf, self.bo]

    def step(self, x, h_, c_):

        # i = tt.tanh(tt.dot(self.Wxi, x) + tt.dot(self.Whi, h_) + self.bi)
        # j = tt.nnet.sigmoid(tt.dot(self.Wxj, x) + tt.dot(self.Whj, h_) + self.bj)
        # f = tt.nnet.sigmoid(tt.dot(self.Wxf, x) + tt.dot(self.Whf, h_) + self.bf)
        # o = tt.nnet.sigmoid(tt.dot(self.Wxo, x) + tt.dot(self.Who, h_) + self.bo)

        xc = tt.concatenate([x,h_])

        i = tt.tanh(tt.dot(self.Wi, xc) + self.bi)
        j = tt.nnet.sigmoid(tt.dot(self.Wj, xc) + self.bj)
        f = tt.nnet.sigmoid(tt.dot(self.Wf, xc) + self.bf)
        o = tt.nnet.sigmoid(tt.dot(self.Wo, xc) + self.bo)

        c = f * c_ + j * i

        h = o * tt.tanh(c)

        return c, h
