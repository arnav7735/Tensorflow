import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Load values from the dataset
x_train = pd.read_csv(r"C:\Users\Arnav\Desktop\TensorFlow\input.csv")
y_train = pd.read_csv(r"C:\Users\Arnav\Desktop\TensorFlow\labels.csv")

x_test = pd.read_csv(r"C:\Users\Arnav\Desktop\TensorFlow\input_test.csv")
y_test = pd.read_csv(r"C:\Users\Arnav\Desktop\TensorFlow\labels_test.csv")

#Label your classes at type int

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Normalize the values from the dataset
x_train = x_train / 255
x_test = x_test / 255

# Convert them into numpy arrays
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# Reshaping to take preferred input sizes

x_train = x_train.reshape(len(x_train), 100, 100, 3)
x_test = x_test.reshape(len(x_test), 100, 100, 3)

# Shuffling of input values is necessary so the model can learn well without any specific pattern in the pictures
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

"""
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
"""

model = keras.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Mention the loss and optimizer functions
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)

plt.imshow(x_test[13])
plt.show()

predictions = model.predict(x_test[13].reshape(1, 100, 100, 3))
predictions = predictions > 0.5

if predictions == 0:
    print("dog")
else:
    print("Cat")
