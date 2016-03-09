import theano as t
import theano.tensor as tt
import numpy as np
import common

class GRU_LAYER(object):

    def __init__(self, init_mean=0, init_var=0, nX=0, nH=0, params=None, name=None):

        if (params != None):

            #load pre trained parameters
            self.Wxr = t.shared(params['W_'+name+'_xr'], name='W_'+name+'_xr', borrow=True)
            self.Wxz = t.shared(params['W_'+name+'_xz'], name='W_'+name+'_xz', borrow=True)
            self.Wxh = t.shared(params['W_'+name+'_xh'], name='W_'+name+'_xh', borrow=True)
            self.Whr = t.shared(params['W_'+name+'_hr'], name='W_'+name+'_hr', borrow=True)
            self.Whz = t.shared(params['W_'+name+'_hz'], name='W_'+name+'_hz', borrow=True)
            self.Whh = t.shared(params['W_'+name+'_hh'], name='W_'+name+'_hh', borrow=True)
            self.br = t.shared(params['b_'+name+'_r'], name='b_'+name+'_r', borrow=True)
            self.bz = t.shared(params['b_'+name+'_z'], name='b_'+name+'_z', borrow=True)
            self.bh = t.shared(params['b_'+name+'_h'], name='b_'+name+'_h', borrow=True)

        else:

            self.Wxr = common.create_weight(input_n=nH, output_n=nX, suffix=name+'_xr')
            self.Wxz = common.create_weight(input_n=nH, output_n=nX, suffix=name+'_xz')
            self.Wxh = common.create_weight(input_n=nH, output_n=nX, suffix=name+'_xh')

            self.Whr = common.create_weight(input_n=nH, output_n=nH, suffix=name+'_hr')
            self.Whz = common.create_weight(input_n=nH, output_n=nH, suffix=name+'_hz')
            self.Whh = common.create_weight(input_n=nH, output_n=nH, suffix=name+'_hh')

            self.br = common.create_bias(output_n=nH, value=0.01, suffix=name+'_r')
            self.bz = common.create_bias(output_n=nH, value=0.01, suffix=name+'_z')
            self.bh = common.create_bias(output_n=nH, value=0.01, suffix=name+'_h')

        self.params = [self.Wxr, self.Wxz, self.Wxh,\
                       self.Whr, self.Whz, self.Whh,\
                       self.br, self.bz, self.bh]

    def step(self, x, h_):

        # xh = tt.concatenate([x,h_])
        #
        # r = tt.nnet.sigmoid(tt.dot(self.Wr, xh) + self.br)
        # z = tt.nnet.sigmoid(tt.dot(self.Wz, xh) + self.bz)
        #
        # xrh = tt.concatenate([x,r*h_])
        # h_hat = tt.tanh(tt.dot(self.Wh, xrh) + self.bh)
        #
        # h = (1-z) * h_ + z * h_hat

        r = tt.nnet.sigmoid(tt.dot(self.Wxr, x) + tt.dot(self.Whr, h_) + self.br)
        z = tt.nnet.sigmoid(tt.dot(self.Wxz, x) + tt.dot(self.Whz, h_) + self.bz)

        h_hat = tt.tanh(tt.dot(self.Wxh, x) + tt.dot(self.Whh, r * h_) + self.bh)
        # h_hat = tt.nnet.relu(tt.dot(self.Wxh, x) + tt.dot(self.Whh, r * h_) + self.bh)

        h = (1-z) * h_ + z * h_hat

        return h
