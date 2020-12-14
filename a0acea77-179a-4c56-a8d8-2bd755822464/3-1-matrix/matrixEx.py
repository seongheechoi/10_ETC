import tensorflow as tf

x = tf.constant([[1.0, 2.0, 3.0]]) # 1 * 3
w = tf.constant([2.0, 2.0, 2.0]) #3,s

print('x shape:', x.get_shape())
print('w shape:', w.get_shape())
w = tf.expand_dims(w, axis=1)
print('w new shape:',w.get_shape())
y = tf.matmul(x, w)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)

print('result:',result)
