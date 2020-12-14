import tensorflow as tf

NUM_STEPS = 5000
MINIBATCH_SIZE = 128

def mnist_load():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.astype('float32') / 255.
    train_y = train_y.astype('int32')
    test_x = test_x.astype('float32') / 255.
    test_y = test_y.astype('int32')
    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = mnist_load()

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_x}, y=train_y, shuffle=True, batch_size=MINIBATCH_SIZE)



feature_columns = [tf.feature_column.numeric_column('x', shape=[28, 28])]

dnn = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[200],
      n_classes=10,
      optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.2),
      model_dir='./model/DNNClassifier'
  )

dnn.train(
      input_fn=train_input_fn,
      steps=NUM_STEPS)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_x}, y=test_y, shuffle=False
  )

test_acc = dnn.evaluate(input_fn=eval_input_fn, steps=1)['accuracy']
print('test accuracy: {}'.format(test_acc))
