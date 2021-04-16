import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)


def f1(synapse, train, var_D, m, j, t):
    train = tf.squeeze(train)
    bb = tf.matmul(tf.reshape(synapse[j, :], (1, m)), tf.reshape(train[:, t], (m, 1)))
    bb_new = tf.reshape(tf.math.subtract(bb, var_D), ())
    if j == 0:
        deta_P = tf.stack([bb_new, 0.0, 0.0], 0)
    if j == 1:
        deta_P = tf.stack([0.0, bb_new, 0.0], 0)
    if j == 2:
        deta_P = tf.stack([0.0, 0.0, bb_new], 0)
    return deta_P


def f2():
    deta_P = tf.constant(0.0, shape=[3], dtype=tf.float32)
    return deta_P


def Leaky_integrate(LAYER2T_P, LAYER2T_T, synapse, train, var_D, m, j, t):
    deta_P = tf.cond(tf.less(LAYER2T_T[j], t), lambda: f1(synapse, train, var_D, m, j, t), lambda: f2())
    LAYER2T_P = tf.add(LAYER2T_P, deta_P)
    return LAYER2T_P, LAYER2T_T


def f3(active_pot):
    img_win = tf.cast(tf.argmax(active_pot), dtype=tf.float32)
    return img_win


def f4(img_win):
    return img_win


def f5_1(ss, LAYER2T_P):
    if ss == 0:
        mult = tf.stack([0.0, 1.0, 1.0], 0)
    if ss == 1:
        mult = tf.stack([1.0, 0.0, 1.0], 0)
    if ss == 2:
        mult = tf.stack([1.0, 1.0, 0.0], 0)
    LAYER2T_P = tf.multiply(LAYER2T_P, mult)
    return LAYER2T_P


def f6_1(ss, LAYER2T_P):
    if ss == 0:
        sub = tf.stack([500.0, 0.0, 0.0], 0)
    if ss == 1:
        sub = tf.stack([0.0, 500.0, 0.0], 0)
    if ss == 2:
        sub = tf.stack([0.0, 0.0, 500.0], 0)
    LAYER2T_P = tf.math.subtract(LAYER2T_P, sub)
    return LAYER2T_P


def f5_2(ss, LAYER2T_T, t_ref):
    if ss == 0:
        add = tf.stack([t_ref, 0.0, 0.0], 0)
    if ss == 1:
        add = tf.stack([0.0, t_ref, 0.0], 0)
    if ss == 2:
        add = tf.stack([0.0, 0.0, t_ref], 0)
    LAYER2T_T = tf.add(LAYER2T_T, add)
    return LAYER2T_T


def f6_2(LAYER2T_T):
    return LAYER2T_T


def Lateral_Inhibition(LAYER2T_P, LAYER2T_T, img_win, t_ref, ss):
    LAYER2T_P = tf.cond(tf.equal(img_win, ss), lambda: f5_1(ss, LAYER2T_P), lambda: f6_1(ss, LAYER2T_P))
    LAYER2T_T = tf.cond(tf.equal(img_win, ss), lambda: f5_2(ss, LAYER2T_T, t_ref), lambda: f6_2(LAYER2T_T))
    return LAYER2T_P, LAYER2T_T


def f9(p, img_win, m):
    crnames = locals()
    cr = tf.expand_dims(tf.constant([0.0, 0.0, 0.0]), 1)

    for i in range(m):
        if i != p:
            cr = tf.expand_dims(tf.constant([0.0, 0.0, 0.0]), 1)
        if i == p:
            if img_win == 0:
                cr = tf.expand_dims(tf.constant([0.4, 0.0, 0.0]), 1)
            if img_win == 1:
                cr = tf.expand_dims(tf.constant([0.0, 0.4, 0.0]), 1)
            if img_win == 2:
                cr = tf.expand_dims(tf.constant([0.0, 0.0, 0.4]), 1)
        if i == 0:
            crnames['cr_' + str(i)] = cr
        else:
            crnames['cr_' + str(i)] = tf.concat([crnames['cr_' + str(i - 1)], cr], axis=1)
    deta_sy = crnames['cr_' + str(m - 1)]

    for k in range(m):
        del crnames['cr_' + str(k)]
    del cr
    return deta_sy


def f10(n, m):
    deta_sy = tf.constant(0.0, shape=[n, m], dtype=tf.float32)
    return deta_sy
