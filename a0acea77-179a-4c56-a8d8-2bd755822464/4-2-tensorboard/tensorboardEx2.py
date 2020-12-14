import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[None, 2], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[None, 2], name="input_b")
    const = tf.constant(2, dtype=tf.float32, name="static_value")

    with tf.name_scope("Transformation"):
        with tf.name_scope("A"):
            a_mul = tf.multiply(in_1, const)
            a_out = tf.subtract(a_mul, in_1)

        with tf.name_scope("B"):
            b_mul = tf.multiply(in_2, const)
            b_out = tf.subtract(b_mul, in_2)

        with tf.name_scope("C"):
            c_div = tf.div(a_out, b_out)
            c_out = tf.add(c_div, const)

        with tf.name_scope("D"):
            d_div = tf.div(b_out, a_out)
            d_out = tf.add(d_div, const)
    out = tf.maximum(c_out, d_out)
    sess = tf.Session()
    _result, _c_out, _d_out = sess.run([out,c_out,d_out],
                            feed_dict={in_1:[[1,2],[3,4]], in_2:[[5,6],[7,8]]})

print('c_out={}, \n\nd_out={}\n\nresult={}'.format(_c_out, _d_out, _result))
tf.summary.FileWriter('./my_graph', graph=graph)
