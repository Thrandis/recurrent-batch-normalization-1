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


def plot(logs, what='error_rate', title='Sequential pMNIST'):
    plt.figure()
    plt.title(title)
    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm', 'b', 'r', 'g', 'k', 'y', 'c', 'm']
    for i, log in enumerate(logs):
        if isinstance(log, tuple):
            name = log[1]
            log = log[0]
        else:
            name = os.path.basename(log)
        data = parse_log(log)
        #plt.subplot(211)
        if what == 'error_rate':
            if len(data['train_error_rate']):
                train = data['train_error_rate']
                valid = data['valid_error_rate']
            else:
                print 'training'
                train = data['train_training_error_rate']
                valid = data['valid_training_error_rate']
            plt.plot(train[:], c=colors[i], ls=':', lw=2)
            plt.plot(valid[:], c=colors[i], lw=2, label=name)
            print len(train)
            print len(valid)
            plt.ylabel('Misc Rate') 
        elif what == 'cross_entropy':
            if len(data['train_cross_entropy']):
                train = data['train_cross_entropy']
                valid = data['valid_cross_entropy']
            else:
                train = data['train_training_cross_entropy']
                valid = data['valid_training_cross_entropy']
            plt.plot(train[:], c=colors[i], ls=':', lw=2)
            plt.plot(valid[:], c=colors[i], lw=2, label=name)
            print len(train)
            print len(valid)
            plt.ylabel('Cross Entropy') 
        plt.legend()
        
    plt.ylim([0.8, 1.8])
    plt.xlim([0, 20])
    plt.grid()
    plt.xlabel('Epochs')
    plt.show()
    #plt.close('all')


logs = ['experiments/results/pmnist/baseline/baseline',
        'experiments/results/pmnist/bn/bn',
        #'experiments/results/pmnist/norm/full_norm_input_gamma=01.txt',
        #'experiments/results/pmnist/norm/norm_tanh_sqrt_full_norm_input_gamma=01.txt',
        'experiments/results/pmnist/norm/full_norm_input_gamma=1.txt',
        'experiments/results/pmnist/norm/norm_tanh_sqrt_full_norm_input_gamma=1.txt',
       ]
#plot(logs)

logs = ['experiments/results/ptb/baseline/baseline',
        'experiments/results/ptb/bn/bn',
        'experiments/results/ptb/norm/rnn_baseline.txt',
        #'experiments/results/ptb/norm/rnn_norm_sqrt_gamma=01.txt',
        #'experiments/results/ptb/norm/rnn_norm_sqrt_gamma=08.txt',
        'experiments/results/ptb/norm/rnn_norm_sqrt_gamma=1.txt',
        #'experiments/results/ptb/norm/rnn_norm_sqrt_norm_tanh_gamma=01.txt',
        'experiments/results/ptb/norm/rnn_norm_sqrt_norm_tanh_gamma=08.txt',
        'experiments/results/ptb/norm/rnn_norm_sqrt_norm_tanh_gamma=1.txt',
       ]
#plot(logs, 'cross_entropy')

logs = ['experiments/results/ptb/baseline/baseline',
        'experiments/results/ptb/bn/bn',
        'experiments/results/ptb/ln/ln',
        #'experiments/results/ptb/norm/rnn_baseline.txt',
        #'experiments/results/ptb/norm/rnn_norm_sqrt_gamma=1.txt',
        #'experiments/results/ptb/norm/lstm_norm_tanh_gamma=01.txt',
        #'experiments/results/ptb/norm/lstm_norm_tanh_c_gamma=01.txt',
        #'experiments/results/ptb/norm/lstm_norm_tanh_c_gamma=01_lr=002.txt',
        ('experiments/results/ptb/norm/lstm_norm_tanh_c_no_force_l2_gamma=01.txt', 'NormProp'),
       ]
plot(logs, 'cross_entropy', 'PenntreeBank')
