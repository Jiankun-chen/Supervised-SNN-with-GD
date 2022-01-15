import tensorflow as tf

def vis_synapse(synapse, pic_size):
    vis_neu0 = tf.reshape(synapse[0, :], [pic_size, pic_size])
    vis_neu0 = tf.expand_dims(vis_neu0, 0)
    vis_neu0 = tf.expand_dims(vis_neu0, -1)
    vis_neu1 = tf.reshape(synapse[1, :], [pic_size, pic_size])
    vis_neu1 = tf.expand_dims(vis_neu1, 0)
    vis_neu1 = tf.expand_dims(vis_neu1, -1)
    vis_neu2 = tf.reshape(synapse[2, :], [pic_size, pic_size])
    vis_neu2 = tf.expand_dims(vis_neu2, 0)
    vis_neu2 = tf.expand_dims(vis_neu2, -1)

    tf.summary.image('vis_0', vis_neu0, max_outputs=4)
    tf.summary.image('vis_1', vis_neu1, max_outputs=4)
    tf.summary.image('vis_2', vis_neu2, max_outputs=4)

