import theano as t
import theano.tensor as tt
import numpy as np
from theano import function
from theano.compile.ops import as_op

def infer_shape(node, input_shapes):
    ashp, bshp = input_shapes
    return [ashp]

@as_op(itypes=[
                tt.fvector, # x_h_
                tt.fvector, # v_h_
                tt.fvector, # t_h_
                tt.fmatrix, # x_t_
                tt.fmatrix, # v_t_
                tt.fvector, # t_t_
                tt.bvector, # exist
               ],
       otypes=[
                tt.fmatrix, # a_t
                tt.fmatrix, # v_t
                tt.fmatrix, # x_t
                tt.fvector, # t_t
                tt.fvector, # t_h
               ])#, infer_shape=infer_shape)

def step(x_h_, v_h_, t_h_, x_t_, v_t_, t_t_, exist):
    '''
    x_h_ : host position in time t-1. (1x2)
    h_h_ : host speed in time t-1. (1x2)
    t_h_ : host waiting time since full stop (1)
    x_t_ : target position in time t-1. (3x2)
    v_t_ : target speed in time t-1. (3x2)
    t_t_ : targets waiting time since full stop (3)
    exist : boolean flag indicating if a car exist in the lane
    '''

    DT = 0.1
    v_t = v_t_ + 1e-6
    x_t = x_t_ + v_t * DT
    t_t = t_t_ + DT
    t_h = t_h_ + DT
    a_t = np.zeros((3,2),dtype=np.float32)

    return a_t, v_t, x_t, t_t, t_h

# x = tt.fmatrix()
# # y = tt.fmatrix()
# #
# f = function([x], my_2pir_jump(x))
# inp1 = 160 * np.random.randn(2,1).astype('float32')
# # inp2 = np.random.rand(2,2).astype('float32')
# out = f(inp1)
# print 'inp1:'
# print inp1
# # print 'inp2:'
# # print inp2
# print 'out:'
# print out