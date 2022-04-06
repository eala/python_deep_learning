import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.utils import np_utils
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()
dims = 28*28
model.add(Dense(units=256,
                input_dim=dims,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# encode value
y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)

# data pre-processing
x_train_2D = x_train.reshape(60000, dims).astype('float32')
x_test_2D = x_test.reshape(10000, dims).astype('float32')

x_train_norm = x_train_2D / 255
x_test_norm = x_test_2D / 255

# training
train_history = model.fit(x = x_train_norm,
                          y = y_train_one_hot,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 800,
                          verbose = 2)

# check model
scores = model.evaluate(x_test_norm, y_test_one_hot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))

# predict
X = x_test_norm[0:10, :]
predictions = np.argmax(model.predict(X), axis = -1)

print(predictions)

# visualize
plt.imshow(x_test[0])
plt.show()
