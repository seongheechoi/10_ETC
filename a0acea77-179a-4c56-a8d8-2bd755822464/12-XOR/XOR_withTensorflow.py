import tensorflow as tf
import numpy as np

tf.set_random_seed(222)
learning_rate = 0.1

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],[1],[1],[0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1') # 2 * 10
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)  #hidden layer output

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2') # 10 * 10
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)  #hidden layer output

W3 = tf.Variable(tf.random_normal([10, 1]), name='weight3') # 10 * 1
b3 = tf.Variable(tf.random_normal([1]), name='bias3')

hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)) #binary crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print('step:{} cost:{}'.format(step, sess.run(cost, feed_dict={X: x_data, Y: y_data})))

    _h, _p, _a = sess.run([hypothesis, prediction, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print('hypothesis:\n{}\nprediction:\n{}\naccuracy:{}'.format(_h, _p, _a))
