import numpy as np
from collections import defaultdict
import matplotlib.pylab as plt
import matplotlib.cm as cm
import sys
import os

plt.rcParams.update({'font.size': 16})
plt.close('all')


def parse_log(file_):
    data = defaultdict(list)
    with open(file_) as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if len(line) == 2:
                try:
                    data[line[0][:-1]].append(float(line[1]))
                except ValueError:
                    continue
    return data


def plot(logs, what='error_rate'):
    plt.figure()
    plt.title('MNIST')
    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm', 'b', 'r', 'g', 'k', 'y', 'c', 'm']
    for i, log in enumerate(logs):
        name = os.path.basename(log)
        data = parse_log(log)
        #plt.subplot(211)
        if what == 'error_rate':
            plt.plot(data['train_error_rate'][:], c=colors[i], ls=':', lw=2)
            plt.plot(data['valid_error_rate'][:], c=colors[i], lw=2, label=name)
            print len(data['train_error_rate'][:])
            print len(data['valid_error_rate'][:])
            plt.ylabel('Misc Rate') 
        elif what == 'cost':
            plt.plot(data['train_sequence_cost'][:], c=colors[i], ls=':', lw=2)
            #plt.plot(data['dev_sequence_cost'][1:-1], c=colors[i], ls='--', lw=2, label=name)
            plt.plot(data['pop_dev_sequence_cost'][:], c=colors[i], lw=2, label=name)
            print len(data['train_sequence_cost'][:-1])
            print len(data['dev_sequence_cost'][1:-1]) 
            plt.ylabel('Cost')
        plt.legend()
        
    #plt.ylim([0, 1])
    #plt.xlim([0, 100])
    plt.grid()
    plt.xlabel('Epochs')
    plt.show()
    #plt.close('all')


logs = ['experiments/sequentialmnist-lstm/log',
        'experiments/sequentialmnist-normlstm/log']
plot(logs)
