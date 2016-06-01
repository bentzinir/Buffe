import theano
from theano import gof

class VectorTimesVector(gof.COp):
    __props__ = ()

    func_file = "./vectorTimesVector.c"
    func_name = "APPLY_SPECIFIC(vector_times_vector)"

    def __init__(self):
        super(VectorTimesVector, self).__init__(self.func_file,
                                                self.func_name)

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 1:
            raise TypeError('y must be a 1-d vector')

        # Create an output variable of the same type as x
        output_var = theano.tensor.TensorType(
                        dtype=theano.scalar.upcast(x.dtype, y.dtype),
                        broadcastable=[False])()

        return gof.Apply(self, [x, y], [output_var])