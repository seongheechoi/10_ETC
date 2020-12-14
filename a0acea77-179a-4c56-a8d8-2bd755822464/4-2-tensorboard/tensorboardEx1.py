import tensorflow as tf

with tf.name_scope('scope_a'):
    a = tf.add(1, 3, name='a_add')
    b = tf.multiply(a, 3, name='b_multiply')

with tf.name_scope('scope_b'):
    c = tf.add(1, 3, name='c_add')
    d = tf.multiply(a, 3, name='d_multiply')

tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())