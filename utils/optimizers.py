
import theano as t
import theano.tensor as tt
import numpy as np
from collections import OrderedDict

def optimizer(lr, param_struct, gradients, solver_params):

    if solver_params['optimizer']=='sgd':
        return sgd(param_struct, gradients, lr)
    elif solver_params['optimizer']=='adadelta':
        return adadelta(param_struct, gradients, solver_params)
    elif solver_params['optimizer']=='adagrad':
        return adagrad(param_struct, gradients, solver_params, lr)
    elif solver_params['optimizer']=='rmsprop':
        return rmsprop(param_struct, gradients, solver_params, lr)
    elif solver_params['optimizer']=='rmsprop_graves':
        return rmsprop_graves(param_struct, gradients, solver_params)
    elif solver_params['optimizer']=='adam':
        return adam(param_struct, gradients, solver_params, lr)

def sgd(param_struct, gradients, lr):

    updates = OrderedDict(( p, p-lr*g ) for p, g in zip( param_struct.params, gradients))

    return updates

def adadelta(param_struct, gradients, solver_params):

    # compute list of weights updates
    updates = OrderedDict()
    for accugrad, accudelta, param, gradient in zip(param_struct._accugrads,
            param_struct._accudeltas, param_struct.params, gradients):

        # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
        agrad = solver_params['rho'] * accugrad + (1 - solver_params['rho']) * gradient * gradient
        dx = - tt.sqrt((accudelta + solver_params['eps'])
                      / (agrad + solver_params['eps'])) * gradient
        updates[accudelta] = (solver_params['rho'] * accudelta
                              + (1 - solver_params['rho']) * dx * dx)
        updates[param] = param + dx
        updates[accugrad] = agrad

    return updates

def adagrad(param_struct, gradients, solver_params, lr):

    # compute list of weights updates
    updates = OrderedDict()
    for accugrad, param, gradient in zip(param_struct._accugrads, param_struct.params, gradients):
        # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
        #agrad = accugrad + gradient * gradient
        agrad = solver_params['rho'] * accugrad + (1 - solver_params['rho']) * gradient * gradient
        dx = - (lr / tt.sqrt(agrad + solver_params['eps'])) * gradient
        updates[param] = param + dx
        updates[accugrad] = agrad

    return updates

def rmsprop(param_struct, gradients, solver_params, lr):

    # compute list of weights updates
    updates = OrderedDict()
    for accugrad, param, gradient in zip(param_struct._accugrads, param_struct.params, gradients):

        agrad = solver_params['rho'] * accugrad + (1 - solver_params['rho']) * gradient * gradient
        dx = - (lr / tt.sqrt(agrad + solver_params['eps'])) * gradient
        updates[param] = param + dx
        updates[accugrad] = agrad

    return updates

def rmsprop_graves(param_struct, gradients, solver_params):

    # compute list of weights updates
    updates = OrderedDict()
    for n_, g_, d_, param, gradient in zip(param_struct._accugrads2, param_struct._accugrads, param_struct._accudeltas, param_struct.params, gradients):

        n = solver_params['aleph'] * n_ + (1 - solver_params['aleph']) * gradient * gradient
        g = solver_params['aleph'] * g_ + (1 - solver_params['aleph']) * gradient
        dx = solver_params['beit'] * d_ - solver_params['gimmel'] * (1 / tt.sqrt(n - g * g + solver_params['dalet'])) * gradient
        updates[param] = param + dx
        updates[g_] = g
        updates[n_] = n
        updates[d_] = dx

    return updates

def adam(param_struct, gradients, solver_params, lr):

    # compute list of weights updates
    updates = OrderedDict()
    time_ = t.shared(np.asarray(0., dtype=np.float32))
    time = time_ + 1
    a = lr * tt.sqrt(1-solver_params['beta_2']**time)/(1-solver_params['beta_1']**time)

    for  g_, n_, param, gradient in zip(param_struct._accugrads, param_struct._accugrads2, param_struct.params, gradients):

        g = solver_params['beta_1']*g_ + (1-solver_params['beta_1'])*gradient
        n = solver_params['beta_2']*g_ + (1-solver_params['beta_2']) * (gradient**2)# * gradient
        dx = - a * g / (tt.sqrt(tt.abs_(n)) + solver_params['eps'])

        updates[param] = param + dx
        updates[g_] = g
        updates[n_] = n
        updates[time_] = time

    return updates