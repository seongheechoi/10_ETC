from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import tensorflow as tf
tf.set_random_seed(111)

df = pd.read_csv('sonar.csv', header=None)

dataset = df.values
x = dataset[:,0:60]
y_data = dataset[:,60]

e = LabelEncoder()
e.fit(y_data)
y = e.transform(y_data)

model = Sequential() #model 객체 생성
model.add(Dense(24,  input_dim=60, activation='relu')) #hidden layer 입력  60 node:24  활성화 relu
model.add(Dense(10, activation='relu'))#두 번째 hidden layer node:10
model.add(Dense(1, activation='sigmoid')) #output layer :1

model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(x, y, epochs=200, batch_size=5) #학습

print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1])) #평가 [0]:loss [1]:accuracy
