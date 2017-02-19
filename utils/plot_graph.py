#!/usr/bin/python


def main(argv):
    import matplotlib.pyplot as plt
    import numpy as np
    for in_file in argv:
        with open(in_file, 'r') as f:
            lines = [l for l in f.read().split('\n') if l.find('R_std') != -1]
            iters = []
            rewards = []
            errors = []
            for l in lines:
                iters.append(float(l.split(' ')[3].replace(',', '')))
                rewards.append(float(l.split(' ')[-3].replace(',', '')))
                errors.append(float(l.split(' ')[-1].replace(',', '')))

        plt.plot(iters, rewards, label=in_file, linewidth=2.0)
        plt.fill_between(iters, np.asarray(rewards)-errors, np.asarray(rewards)+errors, alpha=0.2)
    plt.title('Learning Curves')
    plt.legend(loc=4)
    plt.xlabel('Iterations[x1000]')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
