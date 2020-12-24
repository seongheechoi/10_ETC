import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
tf.set_random_seed(111)

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
kf = KFold(n_splits=10)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]] #0 ~ 6

print(x_data.shape, y_data.shape)

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for train_idx, val_idx in kf.split(x_data, y_data):
        for step in range(500):
            _cost, _a, _= sess.run([cost, accuracy, train], feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                print("step: {:5}\tcost: {:.3f}\taccuracy: {:.2%}".format(step, _cost, _a))

