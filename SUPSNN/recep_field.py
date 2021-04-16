####################################################### README #########################################################

# This file consists of function that convolves an image with a receptive field so that input to the network is
# close to the form perceived by our eyes.

########################################################################################################################
import tensorflow as tf
import numpy as np


def rf_tf(inp):
    sca1 = 0.625
    sca2 = 0.125
    sca3 = -0.125
    sca4 = -0.5

    # Receptive field kernel
    w = [[	sca4, sca3, sca2, sca3, sca4],
         [	sca3, sca2, sca1, sca2, sca3],
         [sca2, sca1, 1, sca1, sca2],
         [	sca3, sca2, sca1, sca2, sca3],
         [	sca4, sca3, sca2, sca3, sca4]]

    filter = tf.convert_to_tensor(w, dtype=tf.float32)
    filter = tf.expand_dims(filter, -1)
    filter = tf.expand_dims(filter, -1)

    pot = tf.nn.conv2d(inp, filter, strides=[1, 1, 1, 1], padding='SAME')

    return pot


def rf_np(inp):
    sca1 = 0.625
    sca2 = 0.125
    sca3 = -0.125
    sca4 = -.5

    # Receptive field kernel
    w = [[	sca4, sca3, sca2, sca3, sca4],
         [	sca3, sca2, sca1, sca2, sca3],
         [  sca2, sca1,  1,   sca1, sca2],
         [	sca3, sca2, sca1, sca2, sca3],
         [	sca4, sca3, sca2, sca3, sca4]]

    pot = np.zeros([inp.shape[0], inp.shape[1]])
    ran = [-2, -1, 0, 1, 2]
    ox = 2
    oy = 2

    # Convolution
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            summ = 0
            for m in ran:
                for n in ran:
                    if (i + m) >= 0 and (i + m) <= inp.shape[0] - 1 and (j + n) >= 0 and (j + n) <= inp.shape[0] - 1:
                        summ = summ + w[ox + m][oy + n] * inp[i + m][j + n] / 255
            pot[i][j] = summ

    return pot
