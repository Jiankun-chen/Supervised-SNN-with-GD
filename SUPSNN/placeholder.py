import tensorflow as tf


def placeholder_inputs(batch_size, m, n, TU):

    img = tf.placeholder(tf.float32, [batch_size, m, TU + 1])
    teaching = tf.placeholder(tf.float32, [batch_size, n, TU])
    is_training = tf.placeholder(tf. bool)

    return img, teaching, is_training
