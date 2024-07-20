import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


model = tf.keras.models.Sequential([

    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')


])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

model.save('my_cifar10_model.h5')


