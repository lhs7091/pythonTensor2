from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_core.python.keras.layers import Flatten
import matplotlib.pyplot as plt
import numpy as np
import random

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = {0:'T-shirt/top',
          1:'Trouser',
          2:'Pullover',
          3:'Dress',
          4:'Coat',
          5:'Sandal',
          6:'Shirt',
          7:'Sneaker',
          8:'Bag',
          9:'Ankle Boot'}

# dataset reshape
# array -> category
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaling
X_train = X_train/255.0
X_test = X_test/255.0

# modeling
model = Sequential()
model.add(Dense(784, input_shape=(28,28,), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#compiling models
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# input dataset in model
model.fit(X_train, y_train, batch_size=200, epochs=1, validation_split=0.2)
'''
48000/48000 [==============================] - 22s 462us/sample - loss: 0.2445 - accuracy: 0.9248 - val_loss: 0.1263 - val_accuracy: 0.9633
'''

# evaluation
score = model.evaluate(X_test, y_test)
print(score)
'''
10000/10000 [==============================] - 2s 202us/sample - loss: 0.1285 - accuracy: 0.9611
[0.12847008485868572, 0.9611]
'''

# real prediction
prediction = model.predict(X_test)
r = random.randint(0, y_test.shape[0])
print('label:', labels[np.argmax(y_test[r])])
print('prediction:', labels[np.argmax(prediction[r])])

plt.imshow(X_test[r].reshape(28,28), cmap='binary')
plt.show()