import tensorflow as tf
import numpy as np
data_set = np.loadtxt('thoraric_surgery.csv', delimiter=',')
x_data = np.array(data_set[:, 0:17], dtype=np.float32)
y_data = np.array(data_set[:, 17], dtype=np.float32)

x = tf.placeholder(dtype=tf.float32, shape=[None, 17])
y = tf.placeholder(dtype=tf.float32, shape=[None])

w1 = tf.Variable(tf.random_normal([17, 30]), name='weight1')
b1 = tf.Variable(tf.random_normal([30]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([30, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([10, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)

#binary crossentropy
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        _cost, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 5 == 0:
            print('step:{} cost:{}'.format(step, _cost))
    _a = sess.run(accuracy, feed_dict={x:x_data, y:y_data})
    print('accuracy:',_a)
