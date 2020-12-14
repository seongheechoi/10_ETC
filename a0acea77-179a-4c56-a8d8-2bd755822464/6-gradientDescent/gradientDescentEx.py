import tensorflow as tf
tf.set_random_seed(111)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(tf.random_normal([1]), name='weight')
x = tf.placeholder(dtype=tf.float32, shape=[3])
y = tf.placeholder(dtype=tf.float32, shape=[3])

hypothesis = x * w
cost = tf.reduce_mean(tf.square(hypothesis - y))

learning_rate = 0.1
'''
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - learning_rate * gradient
update = w.assign(descent)
'''
update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    _cost, _w, _ = sess.run([cost, w, update], feed_dict={x: x_data, y: y_data})
    print('step:{}  cost:{}  w:{}'.format(step, _cost, _w))
