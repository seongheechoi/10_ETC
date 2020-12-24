import tensorflow as tf
import numpy as np
tf.set_random_seed(111)

#요소 - 요소 최소값 / 요소 최대값 - 요소 최소값
def min_max_Scaler(data):
    numeratorData = data - np.min(data, 0)
    denominatorData = np.max(data, 0) - np.min(data, 0)
    return numeratorData / (denominatorData + 1e-7)

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

xy = min_max_Scaler(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    _cost, _h, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
    print('step:{}\ncost:\n{}\nhypothesis:\n{}'.format(step, _cost, _h))
