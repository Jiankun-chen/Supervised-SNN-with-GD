######################################################## README ######################################################

# This file generates rate based spike train from the potential map.

######################################################################################################################
import tensorflow as tf
import numpy as np
from numpy import interp
# from matplotlib import pyplot as plt
import imageio
import math
from parameters import param as par

#import cupy as cp

#deterministic for potential image
def encode(pot):          #与无监督的不匹配，没用上，直接输入的是编码好的spike

    A = tf.constant(6.2753, shape=[pot.shape[0], pot.shape[1]])
    B = tf.constant(3.5644, shape=[pot.shape[0], pot.shape[1]])
    C = tf.constant(600, shape=[pot.shape[0], pot.shape[1]])
    freq = tf.add(tf.multiply(pot, B), A)
    freq1 = tf.floordiv(C, freq)

    # initializing spike train
    train = []

    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):

            temp = np.zeros([(par.T + 1), ])
            # generating spikes according to the firing rate
            k = freq1
            if pot[l][m] > 0:
                while k < (par.T+1):
                    temp[int(k)] = 1
                    k += freq1
            train.append(temp)
            # print(sum(temp))
            # print(temp)
    return train

'''
if __name__ == '__main__':
    # m = []
    # n = []
    img = imageio.imread("training/{}.png".format(1))
    # img = imageio.imread("data/training/0.png")

    pot = rf(img)

    # for i in pot:
    # m.append(max(i))
    # n.append(min(i))

    # print(max(m), min(n))
    # train = encode2(img)
    train = encode(pot)
    f = open('train6.txt', 'w')
    print(np.shape(train))

    for j in range(len(train)):
        for i in range(len(train[j])):
            f.write(str(int(train[j][i])))
        f.write('\n')

    f.close()
'''

