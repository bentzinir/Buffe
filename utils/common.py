import numpy as np
import cPickle
import math
import tensorflow as tf

def detect_nan(node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print '*** NaN detected ***'
            t.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

def get_params(obj):
    params = {}
    for param in obj:
        params[param.name] = param.get_value()
    return params

def dotproduct(v1, v2):
    return sum((a*b) for a,b in zip(v1,v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def save_params(fName,saver,session):
    saver.save(session,fName)

def load_params(fName):
    f = file(fName,'rb')
    obj = cPickle.load(f)
    f.close()
    return obj

def relu(x):
    return 0.5 * (x + abs(x))

def create_lr_func(solver_params):
    if solver_params['lr_type']=='inv':
        return inv_learning_rate
    elif solver_params['lr_type']=='fixed':
        return fixed_learning_rate
    elif solver_params['lr_type']=='episodic':
        return episodic_learning_rate
    else:
        return []

def inv_learning_rate(iter, solver_params):
    return solver_params['base'] * (1 + solver_params['gamma'] * iter) ** (-solver_params['power'])

def fixed_learning_rate(iter, solver_params):
    return solver_params['base']

def episodic_learning_rate(iter, solver_params):
    return solver_params['base'] / (math.floor(iter / solver_params['interval']) + 1)

def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g,w in grads_and_vars:
        tot_grad += tf.abs(tf.reduce_sum(g))
        tot_w += tf.abs(tf.reduce_sum(w))

    return tot_grad/N, tot_w/N
