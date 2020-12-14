import tensorflow as tf
x = tf.constant([[1.], [4.], [7.]]) # 3 * 1
y = tf.constant([[1., 1., 1.]]) # 1 * 3
print('shape:{} {}'.format(x.get_shape(), y.get_shape()))
#shape: (3, 3) (1, 3)
subXY = tf.subtract(x,y)
sess = tf.Session()
result = sess.run(subXY)
print(result)
