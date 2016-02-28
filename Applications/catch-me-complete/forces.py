import theano as t
import numpy as np
import math

def const_force(a, N):
    F = a * np.ones(shape=(N,len(a)), dtype=t.config.floatX)
    return np.float32(F)

def single_sin_force(dt, N, a, b, T):
    F = np.zeros(shape=(N,len(a)), dtype=t.config.floatX)
    time_ = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i,j] = a[j] * math.sin(2 * math.pi * time_ / T[j]) + b[j]
        time_ += dt
    return F

def double_sin_force(dt, N, a1, c1, T1, a2, c2, T2, b):
    F = np.zeros(shape=(N,len(a1)), dtype=t.config.floatX)
    time_ = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i,j] = a1[j] * math.sin( c1 * 2 * math.pi * time_ / T1[j]) + \
               a2[j] * math.sin( c2 *2 * math.pi * time_ / T2[j]) + b[j]
        time_ += dt
    return F

def random_force(a, b, N, dim):
    F = np.random.uniform(low=-a, high=a, size=(N,dim)) + b
    return np.float32(F)
