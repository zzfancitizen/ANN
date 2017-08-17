import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# in_size 输入数 out_size 该层节点数


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer_%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'\\weights', Weights)
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'\\bias', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'\\output', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.sigmoid)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction, name='square'), reduction_indices=[1],
                                        name='reduce_sum'), name='reduce_mean')
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("..\\logs", sess.graph)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # plt.savefig(os.path.abspath('../material') + os.path.sep + 'image_' + str(i) + '.png')
            plt.pause(0.1)

