import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Normalize the data

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
'''



model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
model.save('mnist.h5')
'''
model = tf.keras.models.load_model('mnist.h5')

loss = model.evaluate(x_test, y_test)
print(loss)

plt.imshow(x_test[78])
plt.show()

predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[78:79])
print(predicted_label)














