#!/usr/bin/python


def main(argv):
    import matplotlib.pyplot as plt
    import numpy as np
    for in_file in argv:
        with open(in_file, 'r') as f:
            lines = [l for l in f.read().split('\n') if l.find('R_std') != -1]
            rewards = []
            for l in lines:
                rewards.append(float(l.split(' ')[-9].replace(',', '')))

        plt.plot(np.arange(len(rewards)), rewards, label=in_file)
    plt.title('Learning Curves')
    plt.legend(loc=4)
    plt.xlabel('Iterations[x1000]')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
