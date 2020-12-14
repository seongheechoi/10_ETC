import tensorflow as tf
from sklearn import datasets, preprocessing

boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

NUM_STEPS = 200
MINIBATCH_SIZE = 506

feature_column = [tf.feature_column.numeric_column(key='x', shape=13)]
train_input_fn = tf.estimator.inputs.numpy_input_fn(
          {'x': x_data}, y_data, batch_size=MINIBATCH_SIZE, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
          {'x': x_data}, y_data, batch_size=MINIBATCH_SIZE, shuffle=False)


reg = tf.estimator.LinearRegressor(
          feature_columns=feature_column,
          optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
          model_dir='./model/boston')

reg.train(input_fn=train_input_fn, steps=NUM_STEPS)
MSE = reg.evaluate(input_fn=eval_input_fn, steps=10)

print(MSE)


