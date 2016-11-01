import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_params(fname):
    with open(fname + '/params.json', 'r') as f:
        try:
            data = json.load(f)
        # if the file is empty the ValueError will be thrown
        except ValueError:
            data = {}
    return data

good_params = []
bad_params = []
root_dir = '/home/nir/work/git/Buffe/Applications/mgail/environments/walker'
n_trajs = '04T-'
k = 0
for root, dirs, files in os.walk(root_dir):
    if n_trajs not in root:
        continue
    for file in files:
        if file == 'log_file.txt':
            with open(root + '/' + file) as f:
                lines = [line.split() for line in f if ', R:' in line]
            r = [float(line[-3][:-1]) for line in lines]
            if not r:
                continue

            r.sort()
            max_r = r[-1]
            # load json
            params_i = load_params(root + '/')
            if max_r < 1500:
                bad_params.append(params_i)
            elif max_r > 5000:
                good_params.append(params_i)

bar_width = 0.35
i = 1
B = dict()
G = dict()
for key in bad_params[0].viewkeys():
    b_list = []
    g_list = []
    for p in bad_params:
        b_list.append(str(p[key]))
    for p in good_params:
        g_list.append(str(p[key]))

    B[key] = Counter(b_list)
    G[key] = Counter(g_list)

    b_labels, b_values = zip(*sorted(Counter(b_list).items()))
    g_labels, g_values = zip(*sorted(Counter(g_list).items()))

    b_values = [float(v) / sum(b_values) for v in b_values]
    g_values = [float(v) / sum(g_values) for v in g_values]

    indexes_b = np.arange(len(b_labels))
    indexes_g = np.arange(len(g_labels))

    plt.subplot(4,5,i)
    plt.bar(indexes_b, b_values, bar_width, color='r')
    plt.bar(indexes_g + bar_width, g_values, bar_width, color='b')
    plt.xticks(indexes_b + bar_width, b_labels)
    plt.xticks(indexes_g + bar_width, g_labels)
    plt.title(key)
    i+=1

plt.pause(0.001)
a=1