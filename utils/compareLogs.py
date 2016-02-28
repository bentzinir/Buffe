#!/usr/bin/python

from cProfile import label
import subprocess
import sys
import os
import matplotlib.pyplot as plt

log1 = sys.argv[1]
log2 = sys.argv[2]

LOGS = [log1, log2]
N = len(LOGS)
wdir = os.getcwd()

iters = [None]*N
cost = [None]*N

fig, ax = plt.subplots()

for i in range(N):
    buf = ("cat " + LOGS[i] + "| grep Training: | cut -d ' ' -f5 | sed 's/.$//'")
    ps = subprocess.Popen(buf, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    iters[i] = output.split('\n')

    buf = ("cat " + LOGS[i] + "| grep Training: | cut -d ' ' -f7 | sed 's/.$//'")
    ps = subprocess.Popen(buf, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    cost[i] = output.split('\n')

    ax.plot(iters[i][:-1], cost[i][:-1], label=LOGS[i])

ax.legend(loc='upper center', shadow=True)
plt.xlabel('iters')
plt.ylabel('cost')
plt.ylim(0, 25)
plt.show()
