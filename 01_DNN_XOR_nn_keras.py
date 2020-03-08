import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential

warnings.filterwarnings('ignore')
tf.random.set_seed(777)

# data
x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]
y_data = [[0],
          [1],
          [1],
          [0]]

# define model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(8, activation='tanh', input_shape=(2,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

model.summary()
model.compile(loss="binary_crossentropy", optimizer=optimizer)

# training model
X_train = np.array(x_data)
y_train = np.array(y_data)

model.fit(X_train, y_train, epochs=1000)

score = model.evaluate(X_train, y_train)
print("score : {:.4f}".format(score))

predicted_x = model.predict(X_train)
print(predicted_x)
print(tf.cast(predicted_x>=0.5, dtype=tf.float32))

'''
score.0.0434
[[0.01966098]
 [0.95703065]
 [0.950225  ]
 [0.0571076 ]]
tf.Tensor(
[[0.]
 [1.]
 [1.]
 [0.]], shape=(4, 1), dtype=float32)
'''
