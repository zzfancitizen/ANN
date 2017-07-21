import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(100)
rng = pd.date_range(start='2000', periods =209,freq='M')
ts = pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
# ts.plot(c='b', title="Example")
# plt.show()

TS = np.array(ts)
num_periods = 20  # 10 batches , 20 values per batch
f_horizon = 1  # Forecast Horizon, 1 period into future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1,20,1)

y_data = TS[1:(len(TS)-len(TS) % num_periods)+f_horizon]
y_batches = y_data.reshape(-1,20,1)

def test_data(series, forecast, num_periods):
    test_x_setup = TS[-(num_periods+forecast):]
    testX = test_x_setup[:num_periods].reshape(-1,20,1)
    testY = TS[-(num_periods):].reshape(-1,20,1)
    return testX, testY

X_test, Y_test = test_data(TS, f_horizon, num_periods)

tf.reset_default_graph()

num_periods = 20
inputs = 1
hidden = 100
output = 1

X = tf.placeholder(tf.float32, [None, num_periods, inputs ])
y = tf.placeholder(tf.float32, [None, num_periods, output ])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)   # dynamic instead of static

learning_rate =0.001

stacked_rnn_output = tf.reshape(rnn_output,[-1, hidden])     # changed the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)  # GD
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) # shape of the results

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 2000

with tf.Session() as sess:
    init.run()
    for i in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y:y_batches})
        if i % 100 == 0:
            mse = loss.eval(feed_dict={X:x_batches, y:y_batches})
            print(i, "\tMSE:", mse)
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)

plt.title("Forecast vs Actual")

plt.plot(pd.Series(np.ravel(Y_test)),"bo", markersize=10, label =" Actual")
plt.plot(pd.Series(np.ravel(y_pred)),"r.", markersize=10, label =" Forecast")
plt.legend(loc="lower left")

plt.show()