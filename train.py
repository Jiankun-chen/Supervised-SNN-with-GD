####################################################### README #####################################################

# This is the main file which calls all the functions and trains the network by updating weights

####################################################################################################################
import os
import sys
import numpy as np
import tensorflow as tf
import SUPSNN
from neuron import neuron
import random
from matplotlib import pyplot as plt
import imageio
import scipy.misc
import os
import time as timing
from weight_initialization import learned_weights_synapse
from acc_matrix import winner_count
from acc_matrix import biggest
import heapq
import warnings
import math
from tqdm import trange
import time
from scipy import interpolate
from time import sleep
import logging
import time
from progressbar import *



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)

WORKING_DIR = os.getcwd()
TRAIN_DIR = os.path.join(WORKING_DIR, 'Train')
TEST_DIR = os.path.join(WORKING_DIR, 'Test')
ROOT_LOG_DIR = os.path.join(WORKING_DIR, 'Output')
RUN_NAME = "model"
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
TRAIN_WRITER_DIR = os.path.join(LOG_DIR, 'Train')
TEST_WRITER_DIR = os.path.join(LOG_DIR, 'Test')
CHECKPOINT_FN = 'model.ckpt'
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)
GET_IMGNUM_DIR = os.path.join(TRAIN_DIR, "Spikes")

train_num = len([lists for lists in os.listdir(GET_IMGNUM_DIR) if os.path.isfile(os.path.join(GET_IMGNUM_DIR, lists))])

MAX_STEPS = 3000
BATCH_SIZE = 1
SAVE_INTERVAL = 50
TRAIN_INTERVAL = 50
TEST_INTERVAL = 20


def main():
    # parameters
    T = 70   #70*0.001
    TU = 70
    tinyT = 100  #将时间轴内插成200份,700跑不动
    t_ref = 20   #20*0.001
    pic_size = 128
    m = 128 * 128
    n = 3
    Pref = tf.constant(0*0.001)
    Pmin = tf.constant(-5.0)  #*0.001
    Pth = tf.constant(80.0)  #*0.001
    Prest = tf.constant(0.0)
    D = tf.constant(5.0)  #*0.001
    var_D = 5.0*0.001
    Prest = tf.constant(0.0)
    w_max = 2.  #*0.001

    training_data = SUPSNN.GetData(TRAIN_DIR)
    train_img = training_data.img
    train_label = training_data.labels

    def interp_spike(input, interp_num, m, mode):
        logging.info('Parsing ' + str(mode) + ' membrane potential...')
        (B, W, H) = input.shape
        #T_new = np.linspace(0, T, interp_num)
        img_new = np.zeros((B, W, interp_num + 1))
        progress = ProgressBar()
        for b in progress(range(B)):
            # for b in range(0, B):
            picseq = input[b, :, :]
            # progress = ProgressBar()
            res = []
            # for w in progress(range(W)):
            for w in range(0, W):
                # print(w)
                # print('---------')
                pixseq = picseq[w, :]

                # f = interpolate.interp1d(np.linspace(0,TU,TU+1), pixseq, kind='quadratic')
                # pixseq_new = f(np.linspace(0, TU, interp_num+1))   #对输入脉冲序列插值
                pixseq_interp = np.interp(np.linspace(0, TU, interp_num + 1), np.linspace(0, TU, TU + 1), pixseq)
                pixseq_interp = np.reshape(pixseq_interp, (1, interp_num + 1))
                # print(pixseq_interp.shape)
                # if w == 0:
                # pixseq_temp = pixseq_interp
                # pixseq_temp = np.reshape(pixseq_temp, (1, interp_num+1))
                # res = []
                # else:
                # pixseq_temp = np.reshape(pixseq_temp, (-1, interp_num + 1))
                # print(pixseq_temp.shape)
                # pixseq_temp = np.concatenate((pixseq_temp, pixseq_interp), axis=0)

                # pixseq_temp = np.append(pixseq_temp, pixseq_interp)
                res.append(pixseq_interp.tolist())
                # print('===========','shape=', pixseq_temp.shape)
            picseq_new = np.reshape(np.array(res), (m, interp_num+1))
            img_new[b, :, :] = picseq_new
        return img_new

    def interp_label(input, interp_num, n, mode):
        logging.info('Parsing ' + str(mode) + ' membrane potential...')
        (B, W, H) = input.shape
        label_new = np.zeros((B, W, interp_num + 1))
        progress = ProgressBar()
        for b in progress(range(B)):
            picseq = input[b, :, :]
            res = []
            if mode == 'label':     #label没有0时刻，在每个label的最前面加一位0时刻=0
                labeltime0 = np.zeros(n)
                picseq = np.insert(picseq, 0, values=labeltime0, axis=1)
            for w in range(0, W):
                pixseq = picseq[w, :]
                pixseq_interp = np.interp(np.linspace(0, TU, interp_num + 1), np.linspace(0, TU, TU + 1), pixseq)
                pixseq_interp = np.reshape(pixseq_interp, (1, interp_num + 1))
                res.append(pixseq_interp.tolist())
            picseq_new = np.reshape(np.array(res), (n, interp_num+1))
            label_new[b, :, :] = picseq_new
        return label_new

    train_img_new = interp_spike(input=train_img, interp_num=tinyT, m=m, mode='train')
    train_label_new = interp_label(input=train_label, interp_num=tinyT, n=n, mode='label')
    '''
    img = training_data.img
    (B, W, H) = img.shape
    print(W)
    T_new = np.linspace(0, T, tinyT)
    img_new = np.zeros((B, W, tinyT+1))
    progress = ProgressBar()
    for b in progress(range(B)):
    #for b in range(0, B):
        picseq = img[b, :, :]
        #progress = ProgressBar()
        res = []
        #for w in progress(range(W)):
        for w in range(0, W):
            #print(w)
            #print('---------')
            pixseq = picseq[w, :]

            #f = interpolate.interp1d(np.linspace(0,TU,TU+1), pixseq, kind='quadratic')
            #pixseq_new = f(np.linspace(0, TU, tinyT+1))   #对输入脉冲序列插值
            pixseq_interp = np.interp(np.linspace(0, TU, tinyT+1), np.linspace(0, TU, TU+1), pixseq)
            pixseq_interp = np.reshape(pixseq_interp, (1, tinyT + 1))
            #print(pixseq_interp.shape)
            #if w == 0:
                #pixseq_temp = pixseq_interp
                #pixseq_temp = np.reshape(pixseq_temp, (1, tinyT+1))
                #res = []
            #else:
                #pixseq_temp = np.reshape(pixseq_temp, (-1, tinyT + 1))
                #print(pixseq_temp.shape)
                #pixseq_temp = np.concatenate((pixseq_temp, pixseq_interp), axis=0)

                #pixseq_temp = np.append(pixseq_temp, pixseq_interp)
            res.append(pixseq_interp.tolist())
                #print('===========','shape=', pixseq_temp.shape)
        picseq_new = np.reshape(np.array(res), (16384, 701))
        img_new[b, :, :] = picseq_new
    '''
    #print(train_img_new.shape)
    training_data.img = train_img_new
    training_data.labels = train_label_new

    test_data = SUPSNN.GetData(TEST_DIR)
    test_img = test_data.img
    test_label= test_data.labels

    test_img_new = interp_spike(input=test_img, interp_num=tinyT, m=m, mode='test')
    test_label_new = interp_label(input=test_label, interp_num=tinyT, n=n, mode='label')

    test_data.img = test_img_new
    test_data.labels = test_label_new
    #g = tf.Graph()
    graph = tf.get_default_graph()

    synapse = tf.get_variable('synapse', initializer=tf.random_uniform((n, m), minval=0., maxval=w_max * 0.5, dtype=tf.float32), trainable=True)

    # with g.as_default():
    with graph.as_default():

        SUPSNN.count_flops(graph)

        img, labels, is_training = SUPSNN.placeholder_inputs(batch_size=BATCH_SIZE, m=m, n=n, TU=tinyT)

        train = img

        logging.info('Creating SUP-SNN graph...')

        LAYER2T_P = tf.Variable(tf.constant(0, shape=[n], dtype=tf.float32), name='layer2_potential')

        #LAYER2T_T = tf.get_variable('layer2_time' + "{EPOCH_ID}".format(EPOCH_ID=EPOCH_ID) + "{IMG_ID}".format(IMG_ID=IMG_ID), initializer=tf.constant(0, shape=[n, T], dtype=tf.float32), trainable=True)
        #LAYER2T_T = tf.get_variable('layer2_time', initializer=tf.constant(0, shape=[n, T], dtype=tf.float32), trainable=True)
        LAYER2T_T = tf.Variable(tf.constant(0, shape=[n], dtype=tf.float32), name='layer2_time')

        #variables = tf.global_variables()

        # flag for lateral inhibition
        img_win = tf.constant(100.0)

        active_pot = tf.Variable(tf.zeros([1, n]), trainable=False)

        names = locals()

        for t in range(tinyT+1): #np.float(1*0.001), np.float((T + 1)*0.001), np.float(0.1*0.001)

            for j in range(n):

                LAYER2T_P, LAYER2T_T = SUPSNN.Leaky_integrate(LAYER2T_P, LAYER2T_T, synapse, train, var_D, m, j, t)

            # print(LAYER2T_P.shape)

            active_pot = tf.reshape(LAYER2T_P, (1, n))

            # Lateral Inhibition
            high_pot = tf.maximum(active_pot[0, 0], active_pot[0, 1])
            high_pot = tf.maximum(high_pot, active_pot[0, 2])

            img_win = tf.cond(tf.greater(high_pot, Pth), lambda: SUPSNN.f3(active_pot), lambda: SUPSNN.f4(img_win))

            if img_win == 0 or img_win == 1 or img_win == 2:
                for ss in range(n):

                    LAYER2T_P, LAYER2T_T = SUPSNN.Lateral_Inhibition(LAYER2T_P, LAYER2T_T, img_win, t_ref, ss)

            LAYER2T_P0 = tf.expand_dims(LAYER2T_P, 1)  #为了之后用tf.concat，需要提前增加一个维度

            if t == 0:
                names['temp_' + str(t)] = LAYER2T_P0

            else:
                names['temp_' + str(t)] = tf.concat([names['temp_' + str(t - 1)], LAYER2T_P0], axis=1)

        logits = names['temp_' + str(tinyT)]

        '''
        # Revise synapse for pixel value=0
        for p in range(m):

            print(p)

            row_sum = tf.reduce_sum(tf.squeeze(train), 1)

            deta_sy = tf.cond(tf.equal(row_sum[p], tf.constant(0.0)), lambda: SUPSNN.f9(p, img_win, m), lambda: SUPSNN.f10(n, m))

            synapse = tf.subtract(synapse, deta_sy)

            del deta_sy
        '''

        LOSS_HUBER, LOSS_MSE = SUPSNN.loss_calc(logits=logits, labels=labels)

        total_parameters = SUPSNN.count()

        eq_int = SUPSNN.eq_calc(logits=logits, labels=labels)

        train_op, global_step = SUPSNN.training(loss=LOSS_HUBER, learning_rate=1e-02)

        SUPSNN.vis_synapse(synapse=synapse, pic_size=pic_size)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        # with tf.Session() as sess:

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            #print([x for x in tf.global_variables() if 'Adam' in x.name])

            train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)

            test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

            global_step_value, = sess.run([global_step])

            logging.info("Supervised SNN is Ready, Last Trained Iteration was: " + str(global_step_value))

            for step in range(global_step_value + 1, global_step_value + MAX_STEPS + 1, SAVE_INTERVAL):

                epoch = math.ceil(step / train_num)

                eq_batch_train = 0

                with trange(TRAIN_INTERVAL, ncols=180, ascii='=>', desc='Training', mininterval=0.0005) as ttrain:

                    for sub_step in ttrain:

                        img_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                        train_feed_dict = {img: img_batch,
                                           labels: labels_batch,
                                           is_training: True}

                        _, train_loss_mse_value, train_loss_huber_value, train_eq, train_summary_str = sess.run([train_op, LOSS_MSE, LOSS_HUBER, eq_int, summary], feed_dict=train_feed_dict)

                        eq_batch_train = eq_batch_train + train_eq

                        step_now = step + sub_step

                        train_acc = eq_batch_train / sub_step

                        ttrain.set_postfix(EPOCH=epoch, EPOCH_PIC=sub_step, HUBER_LOSS=train_loss_huber_value, MSE_LOSS=train_loss_mse_value, TRAIN_ACC=train_acc, TOTALITER=step_now)

                    tf.summary.scalar('TRAIN_ACC', train_acc)

                sleep(0.5)

                train_writer.add_summary(train_summary_str, step)

                train_writer.flush()

                eq_batch_test = 0

                with trange(TEST_INTERVAL, ncols=150, ascii='=>', desc='Testing', mininterval=0.0005) as ttest:

                    for sub_step in ttest:

                        img_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                        test_feed_dict = {img: img_batch,
                                          labels: labels_batch,
                                          is_training: False}

                        test_loss_mse_value, test_loss_huber_value, test_eq, test_summary_str = sess.run([LOSS_MSE, LOSS_HUBER, eq_int, summary], feed_dict=test_feed_dict)

                        eq_batch_test = eq_batch_test + test_eq

                        test_acc = eq_batch_test / sub_step

                        ttest.set_postfix(EPOCH=np.float32(1), EPOCH_PIC=np.float32(sub_step), HUBER_LOSS=test_loss_huber_value, MSE_LOSS=test_loss_mse_value, TEST_ACC=test_acc)

                    tf.summary.scalar('TEST_ACC', test_acc)

                test_writer.add_summary(test_summary_str, step)
                test_writer.flush()

                logging.info("Test ACC: " + str(test_acc))

                saver.save(sess, CHECKPOINT_FL, global_step=step)

                logging.info("CHECKPOINT Saved")


if __name__ == '__main__':
    main()
