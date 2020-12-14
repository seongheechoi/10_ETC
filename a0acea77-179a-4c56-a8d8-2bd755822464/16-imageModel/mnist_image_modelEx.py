import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(111)
mnist = input_data.read_data_sets("mnist_data/", one_hot=True) # 784

learning_rate = 0.001
training_epochs = 15
batch_size = 100

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w1 = tf.get_variable('w1', shape=[784,256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.get_variable('w2', shape=[256,256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.get_variable('w3', shape=[256,10],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name='bias3')

logit = tf.matmul(layer2, w3) + b3
hypothesis = tf.nn.softmax(logit)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) # 60000 / 100

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('epoch:{}\tcost:{}'.format(epoch + 1, avg_cost))

print('accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
r = random.randint(0, mnist.test.num_examples - 1)
print("label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("prediction: ", sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x: mnist.test.images[r:r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys')
plt.show()
