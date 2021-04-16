####################################################### README #####################################################

# This is the main file for training and testing

####################################################################################################################
import tensorflow as tf
import SUPSNN
import random
import os
import warnings
import math
from tqdm import trange
from time import sleep
import logging

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

MAX_STEPS = 300000
BATCH_SIZE = 1
SAVE_INTERVAL = 100
TRAIN_INTERVAL = 100
TEST_INTERVAL = 20


def main():
    training_data = SUPSNN.GetData(TRAIN_DIR)
    test_data = SUPSNN.GetData(TEST_DIR)

    graph = tf.get_default_graph()

    # Hyperparameters
    T = 70
    scale = tf.constant(1)
    t_ref = 20
    pic_size = 128
    m = 128 * 128
    n = 3
    Pref = tf.constant(0)
    Pmin = tf.constant(-5.0)
    Pth = tf.constant(80.0)
    Prest = tf.constant(0.0)
    D = tf.constant(5.0)
    TU = T
    var_D = 5.0
    Prest = tf.constant(0.0)
    w_max = 2.
    EPOCH_ID = random.random()
    IMG_ID = random.random()

    synapse = tf.get_variable('synapse', initializer=tf.random_uniform((n, m), minval=0., maxval=w_max * 0.5, dtype=tf.float32), trainable=True)

    with graph.as_default():

        img, labels, is_training = SUPSNN.placeholder_inputs(batch_size=BATCH_SIZE, m=m, n=n, TU=TU)

        train = img

        logging.info('Creating SUP-SNN graph...')

        LAYER2T_P = tf.Variable(tf.constant(0, shape=[n], dtype=tf.float32), name='layer2_potential')

        LAYER2T_T = tf.Variable(tf.constant(0, shape=[n], dtype=tf.float32), name='layer2_time')

        variables = tf.global_variables()

        # flag for lateral inhibition
        img_win = tf.constant(100.0)

        active_pot = tf.Variable(tf.zeros([1, n]), trainable=False)

        names = locals()

        for t in range(1, T + 1):

            for j in range(n):

                LAYER2T_P, LAYER2T_T = SUPSNN.Leaky_integrate(LAYER2T_P, LAYER2T_T, synapse, train, var_D, m, j, t)

            active_pot = tf.reshape(LAYER2T_P, (1, n))

            # Lateral Inhibition
            high_pot = tf.maximum(active_pot[0, 0], active_pot[0, 1])

            high_pot = tf.maximum(high_pot, active_pot[0, 2])

            img_win = tf.cond(tf.greater(high_pot, Pth), lambda: SUPSNN.f3(active_pot), lambda: SUPSNN.f4(img_win))

            for ss in range(n):

                LAYER2T_P, LAYER2T_T = SUPSNN.Lateral_Inhibition(LAYER2T_P, LAYER2T_T, img_win, t_ref, ss)

            LAYER2T_P0 = tf.expand_dims(LAYER2T_P, 1)

            if t == 1:
                names['temp_' + str(t)] = LAYER2T_P0

            else:
                names['temp_' + str(t)] = tf.concat([names['temp_' + str(t - 1)], LAYER2T_P0], axis=1)

        logits = names['temp_' + str(T)]

        LOSS_HUBER, LOSS_MSE = SUPSNN.loss_calc(logits=logits, labels=labels)

        eq_int = SUPSNN.eq_calc(logits=logits, labels=labels)

        train_op, global_step = SUPSNN.training(loss=LOSS_MSE, learning_rate=1e-04)

        SUPSNN.vis_synapse(synapse=synapse, pic_size=pic_size)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)

            test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

            global_step_value, = sess.run([global_step])

            logging.info("Supervised SNN is Ready, Last Trained Iteration was: " + str(global_step_value))

            for step in range(global_step_value + 1, global_step_value + MAX_STEPS + 1, SAVE_INTERVAL):

                epoch = math.ceil(step / train_num)

                eq_batch_train = 0

                with trange(TRAIN_INTERVAL, ncols=180, ascii='=>', desc='Training', mininterval=0.0005) as t:

                    for sub_step in t:

                        img_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                        train_feed_dict = {img: img_batch,
                                           labels: labels_batch,
                                           is_training: True}

                        _, train_loss_mse_value, train_loss_huber_value, train_eq, train_summary_str = sess.run([train_op, LOSS_MSE, LOSS_HUBER, eq_int, summary], feed_dict=train_feed_dict)

                        eq_batch_train = eq_batch_train + train_eq

                        step_now = step + sub_step

                        train_acc = eq_batch_train / sub_step

                        t.set_postfix(EPOCH=epoch, EPOCH_PIC=sub_step, HUBER_LOSS=train_loss_huber_value, MSE_LOSS=train_loss_mse_value, TRAIN_ACC=train_acc, TOTALITER=step_now)

                    tf.summary.scalar('TRAIN_ACC', train_acc)

                sleep(0.5)

                train_writer.add_summary(train_summary_str, step)

                train_writer.flush()

                eq_batch_test = 0

                with trange(TEST_INTERVAL, ncols=150, ascii='=>', desc='Testing', mininterval=0.0005) as t:

                    for sub_step in t:

                        img_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                        test_feed_dict = {img: img_batch,
                                          labels: labels_batch,
                                          is_training: False}

                        test_loss_mse_value, test_loss_huber_value, test_eq, test_summary_str = sess.run([LOSS_MSE, LOSS_HUBER, eq_int, summary], feed_dict=test_feed_dict)

                        eq_batch_test = eq_batch_test + test_eq

                        test_acc = eq_batch_test / sub_step

                        t.set_postfix(EPOCH=1, EPOCH_PIC=sub_step, HUBER_LOSS=test_loss_huber_value, MSE_LOSS=test_loss_mse_value, TEST_ACC=test_acc)

                    tf.summary.scalar('TEST_ACC', test_acc)

                test_writer.add_summary(test_summary_str, step)
                test_writer.flush()

                logging.info("Test ACC: " + str(test_acc))

                saver.save(sess, CHECKPOINT_FL, global_step=step)

                logging.info("CHECKPOINT Saved")


if __name__ == '__main__':
    main()
