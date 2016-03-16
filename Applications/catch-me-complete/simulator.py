import theano as t
import theano.tensor as tt
import numpy as np
from theano import function
from theano.compile.ops import as_op

def infer_shape(node, input_shapes):
    ashp, bshp = input_shapes
    return [ashp]

@as_op(itypes=[
                tt.fvector, # x_target_
                tt.fvector, # v_target_
               ],
       otypes=[
                tt.fvector, # x_target
                tt.fvector, # v_target
               ])

def step(x_target_, v_target_):
    '''
    x_host_ : host position in time t-1. (1x2)
    v_host_ : host velocity in time t-1. (1x2)
    x_target_ : target position in time t-1. (1x2)
    v_target_ : target velocity in time t-1. (1x2)
    u_ext : external force (1x2)
    u : applied force (1x2)
    '''

    def bounce(x,v,w):
        if x[0]<0:
            x[0] = -x[0]
            v[0] = - v[0]
        if x[0]>w:
            dx = x[0] - w
            x[0] = w - dx
            v[0] = - v[0]
        if x[1]<0:
            x[1] = -x[1]
            v[1] = - v[1]
        if x[1]> w:
            dy = x[1]-w
            x[1] = w - dy
            v[1] = - v[1]
        return x, v+1e-6

    # constants
    W = 60
    DT = 0.1

    # step target
    x_target = x_target_ + v_target_ * DT
    x_target, v_target = bounce(x_target, v_target_, W)

    return x_target, v_target