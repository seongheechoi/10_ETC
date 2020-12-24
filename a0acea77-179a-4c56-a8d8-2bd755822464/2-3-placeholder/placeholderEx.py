import tensorflow as tf
a = tf.placeholder(dtype=tf.float32, shape=[2,2])
b = tf.reduce_prod(a) # 2차원 데이터 전달 누적곱 (5*4) * (3*7) = 420
c = tf.reduce_sum(a)  # 2차원 데이터 전달 누적합 (5+4) + (3+7) = 19
d = tf.add(b, c)

sess = tf.Session()
print(sess.run([b,c,d], feed_dict={a:[[5,3],[4,7]]}))
