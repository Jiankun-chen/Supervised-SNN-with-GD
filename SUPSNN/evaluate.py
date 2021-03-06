import tensorflow as tf

def loss_calc(logits, labels):

    labels = tf.cast(labels, dtype=tf.float32)
    label = tf.squeeze(labels)
    #label_t0 = tf.constant([[0], [0], [0]], dtype=tf.float32)
    #label = tf.concat([label_t0, label], axis=1)
    label_0 = label[0, :]
    label_1 = label[1, :]
    label_2 = label[2, :]
    logit_0 = logits[0, :]
    logit_1 = logits[1, :]
    logit_2 = logits[2, :]

    #mse_0 = tf.losses.mean_squared_error(label_0, logit_0)
    #mse_1 = tf.losses.mean_squared_error(label_1, logit_1)
    #mse_2 = tf.losses.mean_squared_error(label_2, logit_2)

    #huber_0 = tf.losses.huber_loss(label_0, logit_0, weights=1.0, delta=1.0)
    #huber_1 = tf.losses.huber_loss(label_1, logit_1, weights=1.0, delta=1.0)
    #huber_2 = tf.losses.huber_loss(label_2, logit_2, weights=1.0, delta=1.0)

    #ee = tf.reduce_sum(synapse)
    #LOSS_MSE = (mse_0+mse_1+mse_2)/3.0
    #LOSS_HUBER = (huber_0+huber_1+huber_2)/3.0

    #labels = tf.cast(tf.squeeze(labels), dtype=tf.float32)
    #label_00 = tf.reduce_sum(labels[0, :], 0)

    #通过label获取类别真值
    label_00 = tf.reduce_sum(label[0, :])
    #label_11 = tf.reduce_sum(labels[1, :], 0)
    label_11 = tf.reduce_sum(label[1, :])
    #label_22 = tf.reduce_sum(labels[2, :], 0)
    label_22 = tf.reduce_sum(label[2, :])
    temp1 = tf.cond(tf.greater(label_00, label_11), lambda: 0.0, lambda: 1.0)
    gt_cl = tf.cond(tf.greater(temp1, label_22), lambda: temp1, lambda: 2.0)

    def f_a(label, logit):
        huber = tf.losses.huber_loss(label, logit, weights=1.0, delta=1.0)
        return huber

    def f_b():
        huber = tf.cond(tf.equal(gt_cl, 1), lambda: f_a(label_1, logit_1), lambda: f_a(label_2, logit_2))
        return huber

    '''
    def f_c(label, logit):
        huber = tf.losses.huber_loss(label, logit, weights=1.0, delta=1.0)
        return huber

    def f_d(label, logit):
        huber = tf.losses.huber_loss(label, logit, weights=1.0, delta=1.0)
        return huber
    '''
    huber = tf.cond(tf.equal(gt_cl, 0), lambda: f_a(label_0, logit_0), lambda : f_b())

    LOSS_HUBER = huber

    def f_e(label, logit):
        mse = tf.losses.mean_squared_error(label, logit)
        return mse

    def f_f():
        mse = tf.cond(tf.equal(gt_cl, 1), lambda: f_e(label_1, logit_1), lambda: f_e(label_2, logit_2))
        return mse

    '''
    def f_g(label, logit):
        mse = tf.losses.mean_squared_error(label, logit)
        return mse

    def f_h(label, logit):
        mse = tf.losses.mean_squared_error(label, logit)
        return mse
    '''
    mse = tf.cond(tf.equal(gt_cl, 0), lambda: f_e(label_0, logit_0), lambda : f_f())

    LOSS_MSE = mse

    tf.summary.scalar('LOSS_HUBER', LOSS_HUBER)
    tf.summary.scalar('LOSS_MSE', LOSS_MSE)

    return LOSS_HUBER, LOSS_MSE


def eq_calc(logits, labels):
    labels = tf.cast(tf.squeeze(labels), dtype=tf.float32)

    label_0 = tf.reduce_sum(labels[0, :], 0)
    label_1 = tf.reduce_sum(labels[1, :], 0)
    label_2 = tf.reduce_sum(labels[2, :], 0)
    temp1 = tf.cond(tf.greater(label_0, label_1), lambda: 0.0, lambda: 1.0)
    gt_cl = tf.cond(tf.greater(temp1, label_2), lambda: temp1, lambda: 2.0)

    logits = tf.cast(logits, dtype=tf.float32)
    logits_0 = tf.reduce_sum(logits[0, :], 0)
    logits_1 = tf.reduce_sum(logits[1, :], 0)
    logits_2 = tf.reduce_sum(logits[2, :], 0)
    temp2 = tf.cond(tf.greater(logits_0, logits_1), lambda: 0.0, lambda: 1.0)
    pre_cl = tf.cond(tf.greater(temp2, logits_2), lambda: temp2, lambda: 2.0)
    eq = tf.equal(pre_cl, gt_cl)
    eq_int = tf.to_int32(eq)
    return eq_int

'''
def winner_count(img_win, count1, count2, count3):
    def f50(count1):
        count1 = count1 + 1
        return count1
    def f51(count2):
        tf.cond(tf.equal(img_win, tf.constant(0.0)), lambda: f50(count1), lambda: f51(count2))
        count2 = count2 + 1
        return count2

    t = tf.cond(tf.equal(img_win, tf.constant(0.0)), lambda: f50(count1), lambda: f51(count2))
'''

def evaluation(logits, labels):
    # labels1 = tf.one_hot(labels, 6, axis=-1)
    # labels1 = tf.cast(labels1, tf.int64)
    # labels1 = tf.one_hot(labels, 3)
    # labels1 = tf.cast(labels1, tf.int64)
    # labels1 = labels1[...,0]
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    out_label1 = tf.argmax(logits, 3)
    out_label = tf.transpose(out_label1, perm=[1, 2, 0])
    return accuracy, out_label