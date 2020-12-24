import tensorflow as tf
var = tf.Variable(2, dtype=tf.int32)
update = tf.assign(var, 4)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print('var:',sess.run(var))
sess.run(update)
print('var:',sess.run(var))
