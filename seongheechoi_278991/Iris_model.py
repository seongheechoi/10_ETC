import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def mean_squared_error(y, t):     # MSE 오차함수
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):    # 오차함수 활성화 함수를 소프트 맥스 사용할 경우
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))

tf.set_random_seed(777)
Iris_data = pd.read_csv('iris.data', names=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class'])
data_set = Iris_data.values       # 데이터프레임에서 배열로

with tf.name_scope('input_layer'):     # input layer 로 블럭
    x_data = data_set[:,:4]
    y_data = data_set[:,4]

    e = LabelEncoder()
    e.fit(y_data)      #레이블 인코더를 이용하여 문자열을 int타입으로 변경
    y_data = e.transform(y_data)      #150 형태
    y_data = np.expand_dims(y_data, axis=1)  #150,1 형태

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=2)   #trianing 데이터와 test 데이터로 구분 25%
    # print(x_train.shape)      #결과 (112,4)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    y_one_hot = tf.one_hot(y, 3)     # shape (?,1,3)
    y_one_hot = tf.reshape(y_one_hot, [-1, 3])     # reshape(?,3)

with tf.name_scope('hidden_layer'):
    w1 = tf.Variable(tf.random_normal([4, 15]), name='weight1')
    b1 = tf.Variable(tf.random_normal([15]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)

    w1_hist = tf.summary.histogram("weight1", w1)

with tf.name_scope('output_layer'):
    w2 = tf.Variable(tf.random_normal([15, 3]), name='weight2')
    b2 = tf.Variable(tf.random_normal([3]), name='bias2')

    w2_hist = tf.summary.histogram("weight2", w2)

    logit = tf.matmul(layer1,w2) + b2    #가중의 합
    hypothesis = tf.nn.softmax(logit)    #소프트맥스 활성화 함수

with tf.name_scope('Optimizer'):
    cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                    labels=y_one_hot)  #3개의 오류값, 소프트맥스+크로스앤트로피 적용
    cost = tf.reduce_mean(cost_i)     #평균 오류값

    train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
    #[[0.2, 0.6, 0.2]]
    correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(y_one_hot, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    accuracy_scalar = tf.summary.scalar("accuracy", accuracy)
    cost_scalar = tf.summary.scalar("cost", cost)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./my_graph")
    writer.add_graph(sess.graph)  # Show the graph
    sess.run(tf.global_variables_initializer())
    for step in range(3001):
        summary, _cost, _, acc = sess.run([merged_summary, cost, train, accuracy], feed_dict={x:x_train, y:y_train})
        writer.add_summary(summary, global_step=step)
        if step % 100 == 0:
            print('step:{} cost:{} accuracy:{}'.format(step, _cost, acc))
    _a1 = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
    _a2 = sess.run(accuracy, feed_dict={x:x_test, y:y_test})
    print('accuracy_train:{} accuracy_test:{}'.format(_a1, _a2))

#tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())
#tf.summary.scalar('cost(cross_entropy)', cost)
#tf.summary.histogram('W1',w1)
#tf.summary.histogram('W2',w2)