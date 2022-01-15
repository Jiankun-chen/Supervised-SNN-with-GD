import tensorflow as tf

def placeholder_inputs(batch_size, m, n, TU):

    img = tf.placeholder(tf.float32, [batch_size, m, TU+1])
    label = tf.placeholder(tf.float32, [batch_size, n, TU+1])
    is_training = tf.placeholder(tf. bool)
    '''
    names = locals()
    for tt in range(1, TU + 1):
        for jj in range(n):
            names['op_a_' + str(tt) + '_' + str(jj)] = tf.placeholder(tf.float32, [n, TU])
            names['op_bc_' + str(tt) + '_' + str(jj)] = tf.placeholder(tf.float32, [n, TU])
            names['op_de_' + str(tt) + '_' + str(jj)] = tf.placeholder(tf.float32, [n, TU])
    '''
    return img, label, is_training
