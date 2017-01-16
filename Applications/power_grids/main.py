import tensorflow as tf
import numpy as np
from root_finder import ROOT_FINDER
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

# TYPE='POLYNOMIAL'
TYPE = 'TRIGONOMETRIC'


def el_madad(f, params, roots_p, x_range, n_samples=20):
    sampled_x = np.random.uniform(low=x_range[0], high=x_range[1], size=n_samples)
    madad = []
    for p, (r1, r2) in zip(params, roots_p):
        if TYPE == 'POLYNOMIAL':
            num = np.mean([abs(f(r1, p)), abs(f(r2, p))])
            den = []
            for x_sample in sampled_x:
                den.append(f(x_sample, p))

            madad.append(num / (max(den) - min(den)))

        elif TYPE == 'TRIGONOMETRIC':
            num = f(np.asarray([r1, r2]), p)[0]
            madad.append(abs(num/p[0]))


    return (np.asarray(madad)).mean()


def f(x, *args):
    params = args[0]
    if TYPE == 'POLYNOMIAL':
        return np.dot(params, np.asarray([x**2, x, 1]))
    elif TYPE == 'TRIGONOMETRIC':
        # return -params[0] + params[1]*np.sin(x[0]) + params[2]*np.sin(x[1]), 0
        return -params[0] + params[1] * np.square(x[0]) + params[2] * np.abs(x[1]), 0

def fprime(x):
    return params[1]*np.cos(x[0]) + params[2]*np.cos(x[1])


def get_root(params, x_dim):
    a = params[0]
    b = params[1]
    c = params[2]

    if TYPE == 'POLYNOMIAL':
        t = b**2 - 4 * a * c
        if t >= 0:
            disc = np.sqrt(t)
            roots = np.asarray([(1/(2*a))*(-b - disc), (1/(2*a))*(-b + disc)])
            return True, roots
        else:
            return False, []
    elif TYPE == 'TRIGONOMETRIC':
        sol = fsolve(f, x0=np.random.normal(size=x_dim), args=params)
        if abs(np.asarray(f(sol, params))).sum() < 1e-4:
            valid = True
        else:
            valid = False

        return valid, sol


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def draw_params(n_params, x_dim, batch_size):
    params_batch = []
    roots_batch = []
    for i in xrange(batch_size):
        valid = False
        while not valid:
            if TYPE == 'POLYNOMIAL':
                a = np.random.uniform(low=1, high=2, size=1)
                b = np.random.uniform(low=-0.2, high=0.2, size=1)
                c = np.random.uniform(low=-0.5, high=0, size=1)
                params = np.concatenate([a, b, c])
            elif TYPE == 'TRIGONOMETRIC':
                params = np.random.uniform(low=0, high=0.5, size=n_params)

            valid, roots = get_root(params, x_dim)

        params_batch.append(params)
        roots_batch.append(roots)
    return np.asarray(params_batch), np.asarray(roots_batch)


# 0. problem definition: calculate roots of a parameterized 2nd order polinom: ax^2 + bx + c

# 1. define graph (Feed-forward network)
n_params = 3
n_roots = 2
x_dim = 2
n_iters = 1000000
net_size = [25, 25]
batch_size = 32

params = tf.placeholder("float", shape=(batch_size, n_params))
roots = tf.placeholder("float", shape=(batch_size, n_roots))
dropout_keep = tf.placeholder("float", shape=())


root_finder = ROOT_FINDER(in_dim=n_params, out_dim=n_roots, size=net_size, do_keep_prob=dropout_keep, lr=0.000001)

roots_prediction = root_finder.forward(params)

loss = tf.reduce_mean(tf.square(roots - roots_prediction))

apply_grads, mean_abs_grad, mean_abs_w = root_finder.backward(loss)

init_graph = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_graph)

loss_vec = []
madad_vec = []
iters_vec = []
fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_autoscale_on(True)
axes.autoscale_view(True,True,True)

l1, = axes.plot([], [], label='el madad')
l2, = axes.plot([], [], label='loss')
plt.legend()
plt.show(block=False)
plt.autoscale(enable=True, axis='both', tight=None)

# 2. Train loop
for i in xrange(n_iters):
    params_, roots_ = draw_params(n_params, x_dim, batch_size)
    run_vals = sess.run(fetches=[apply_grads, loss, roots_prediction, mean_abs_grad, mean_abs_w], feed_dict={params: params_, roots: roots_, dropout_keep: 1.})
    roots_p = run_vals[2]

    if i % 1000 == 0:
        madad = el_madad(f, params_, roots_p, x_range=[-2, 2])
        madad_vec.append(madad)
        loss_vec.append(run_vals[1])
        iters_vec.append(i)
        print 'Processed %d/%d iters. Loss: %.4f, madad: %.4f, abs_grad: %.2f, abs_w: %.2f, roots_mean: %.2f, roots_std: %.2f' %\
              (i, n_iters, run_vals[1], madad, run_vals[3], run_vals[4], roots_.mean(), roots_.std())
        l1.set_data(iters_vec, madad_vec)
        l2.set_data(iters_vec, loss_vec)
        axes.relim()
        axes.autoscale_view(True, True, True)
        plt.draw()
        plt.autoscale(enable=True, axis='both', tight=None)
        time.sleep(0.001)

