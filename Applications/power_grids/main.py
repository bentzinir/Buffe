import tensorflow as tf
import numpy as np
import sys, os
from root_finder import ROOT_FINDER
import matplotlib.pyplot as plt

# print os.path.dirname(os.path.realpath(__file__))
# print sys.path
def get_root(params):
    a = params[0]
    b = params[1]
    c = params[2]
    t = b**2 - 4 * a * c
    if t >= 0:
        disc = np.sqrt(t)
        roots = np.asarray([(1/(2*a))*(-b - disc), (1/(2*a))*(-b + disc)])
        # roots = np.asarray([(-b - disc),(-b + disc)])
        return True, roots
    else:
        return False, []

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def draw_params(n_params):
    valid = False
    while not valid:
        params = np.random.uniform(low=1, high=2, size=(1, n_params))
        params[0][1] = 6 * params[0][1]
        # params = np.asarray([[1., 3., 2.]], dtype=np.float32)
        valid, roots = get_root(params[0])
    return params, roots


# 0. problem definition: calculate roots of a parameterized 2nd order polinom: ax^2 + bx + c

# 1. define graph (Feed-forward network)
n_params = 3
n_roots = 2
n_iters = 1000000
net_size = [25, 25]

params = tf.placeholder("float", shape=(1, n_params))
roots = tf.placeholder("float", shape=n_roots)
dropout_keep = tf.placeholder("float", shape=())


root_finder = ROOT_FINDER(in_dim=n_params, out_dim=n_roots, size=net_size, do_keep_prob=dropout_keep, lr=0.000001)

roots_prediction = root_finder.forward(params)

loss = tf.nn.l2_loss(roots - roots_prediction)

apply_grads, mean_abs_grad, mean_abs_w = root_finder.backward(loss)

init_graph = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_graph)

run_vals_vec = np.zeros(n_iters)
# 2. Train loop
for i in xrange(n_iters):
    params_, roots_ = draw_params(n_params)
    run_vals = sess.run(fetches=[apply_grads, loss, mean_abs_grad, mean_abs_w], feed_dict={params: params_, roots: roots_, dropout_keep: 1.})
    run_vals_vec[i] = run_vals[1]
    if i % 1000 == 0:
        print 'Processed %d/%d iters. Loss: %.4f, abs_grad: %.2f, abs_w: %.2f' % (i, n_iters, run_vals[1], run_vals[2], run_vals[3])

plt.plot(moving_average(run_vals_vec))
plt.show(block=True)