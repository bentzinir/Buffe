import theano as t
import theano.tensor as tt
import numpy as np
from theano import function
from theano.compile.ops import as_op

def infer_shape(node, input_shapes):
    ashp, bshp = input_shapes
    return [ashp]

@as_op(itypes=[tt.fmatrix],
       otypes=[tt.fmatrix])#, infer_shape=infer_shape)

def my_2pir_jump(x):
    x_out = np.zeros_like(x)
    for idx,val in enumerate(x):
        if val[0] < 0 :
            x_out[idx] = val[0] + 62.831853 # 2*pi*r
        else:
            x_out[idx] = val[0]
    return x_out

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