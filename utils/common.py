import cPickle
import math
import time
import tensorflow as tf
import numpy as np
import os

def get_params(obj):
    params = {}
    for param in obj:
        params[param.name] = param.get_value()
    return params


def dotproduct(v1, v2):
    return sum((a*b) for a,b in zip(v1,v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def save_params(fname, saver, session):
    saver.save(session, fname)


def load_params(fName):
    f = file(fName,'rb')
    obj = cPickle.load(f)
    f.close()
    return obj


def create_lr_func(solver_params):
    if solver_params['lr_type'] == 'inv':
        return inv_learning_rate
    elif solver_params['lr_type'] == 'fixed':
        return fixed_learning_rate
    elif solver_params['lr_type'] == 'episodic':
        return episodic_learning_rate
    else:
        return []


def inv_learning_rate(itr, solver_params):
    return solver_params['base'] * (1 + solver_params['gamma'] * itr) ** (-solver_params['power'])


def fixed_learning_rate(itr, solver_params):
    return solver_params['base']


def episodic_learning_rate(itr, solver_params):
    return solver_params['base'] / (math.floor(itr / solver_params['interval']) + 1)


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g,w in grads_and_vars:
        tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))

    return tot_grad/N, tot_w/N


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform_initializer(shape):
    scale = np.sqrt(6. / sum(get_fans(shape)))
    weight = (np.random.uniform(-1, 1, size=shape) * scale).astype(np.float32)
    return weight


def print_tensor_stat(x, name):
    x_mean = tf.reduce_mean(x)
    x_std = tf.reduce_mean(tf.square(x - x_mean))
    x_max = tf.reduce_max(x)
    x = tf.Print(x, [x_mean, x_std, x_max], message=name)
    return x


def calculate_gamma(itrvl, gamma_f, t):
    return np.clip(0.1 * int(t/itrvl), 0, gamma_f)


def multivariate_pdf_tf(x, mu, sigma):
    inv_sqrt_2pi = (1. / np.sqrt(2 * np.pi)).astype(np.float32)
    A = tf.mul(inv_sqrt_2pi, tf.div(1., sigma))
    B = tf.reduce_sum(tf.mul(tf.square(x - mu), sigma), reduction_indices=[1])
    p_x = tf.mul(A, tf.exp(tf.mul(-0.5, B)))
    p_x = tf.stop_gradient(p_x)
    return p_x


def multivariate_pdf_np(x, mu, sigma):
    inv_sqrt_2pi = (1. / np.sqrt(2 * np.pi)).astype(np.float32)
    A = inv_sqrt_2pi / sigma
    B = ((x - mu) ** 2 * sigma).sum()
    p_x = A * np.exp(-0.5 * B)
    return p_x


def save_er(module, directory):
    fname = directory + '/er' + time.strftime("-%Y-%m-%d-%H-%M") + '.bin'
    f = file(fname, 'wb')
    cPickle.dump(module, f)
    print 'saved ER: %s' % fname


def load_er(fname, batch_size, history_length, traj_length):
    f = file(fname, 'rb')
    object = cPickle.load(f)
    object.batch_size = batch_size
    state_dim = object.states.shape[-1]
    action_dim = object.actions.shape[-1]
    object.prestates = np.empty((batch_size, history_length, state_dim), dtype=np.float32)
    object.poststates = np.empty((batch_size, history_length, state_dim), dtype=np.float32)
    object.traj_states = np.empty((batch_size, traj_length, state_dim), dtype=np.float32)
    object.traj_actions = np.empty((batch_size, traj_length-1, action_dim), dtype=np.float32)
    return object


def scalar_to_4D(x):
    return tf.expand_dims(tf.expand_dims(tf.expand_dims(x, -1), -1), -1)


def add_field(self, name, size):
    setattr(self, name + '_field', np.arange(self.f_ptr, self.f_ptr + size))
    self.f_ptr += size


def compile_modules(run_dir):
    cwd = os.getcwd()
    os.chdir(run_dir)
    os.system('g++ -std=c++11 simulator.c -o simulator')
    os.system('g++ -std=c++11 -shared pipe.cc -o pipe.so -fPIC -I $TF_INC')
    os.chdir(cwd)


def relu(x, alpha=1./5.5):
    return tf.maximum(alpha * x, x)


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu


def logfunc(x, x2):
    return tf.mul(x, tf.log(tf.div(x, x2)))


def kl_div(rho, rho_hat):
    invrho = tf.sub(tf.constant(1.), rho)
    invrhohat = tf.sub(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho


def normalize(x, mu, std):
    return (x - mu)/std


def denormalize(x, mu, std):
    return x * std + mu
