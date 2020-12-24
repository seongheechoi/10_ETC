import tensorflow as tf
a = tf.constant([5], dtype = tf.float32)
b = tf.constant([10], dtype = tf.float32)
c = tf.constant([2], dtype = tf.float32)
d = a*b+c
sess = tf.Session()
result = sess.run(d)
print('result:',result)

e = tf.multiply(a, b) # a, b노드 곱하기 노드 구성
f = tf.add(e, c) # e, c 더하기 연산 노드 구성
h = tf.add(tf.multiply(a,b), c)
result2 = sess.run(h)
print('result2:', result2)