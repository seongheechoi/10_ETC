import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])
sess.run(queue1.size())
enque_op.run()
print('queue size:',sess.run(queue1.size()))


enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

print('queue size:',sess.run(queue1.size()))


x = queue1.dequeue()
print(x.eval())
print(x.eval())
print(x.eval())
print(x.eval())
print(x.eval())
