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
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6)
model.save('mnist.h5')

'''

model = tf.keras.models.load_model('mnist.h5')


plt.imshow(x_test[0])
plt.show()

predictions = model.predict(x_test)
print(np.argmax(predictions[0:1]))


path = r"C:\Users\Arnav\Desktop\Tensorflow2.0\mnist\working numbers\Untitled.png"

# Load and preprocess the custom image
image_1 = Image.open(path).convert('L')  # Convert to grayscale
image_1 = image_1.resize((28, 28))  # Resize to 28x28

# Display the custom image before normalization
plt.imshow(image_1, cmap='gray')
plt.title("Custom image before normalization")
plt.show()

# Convert image to numpy array and normalize
image_2 = np.array(image_1)  # Convert to numpy array
image_2 = 1-(image_2 / 255.0) # Normalize to [0, 1]


# Display pixel values for comparison
print("Sample MNIST image pixel values:", x_test[0].flatten()[:100])  # Print first 100 pixel values
print("Custom image pixel values:", image_2.flatten()[:100])  # Print first 100 pixel values

# Reshape to match the model input shape
image_2 = image_2.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)

# Display the custom image after normalization and reshaping
plt.imshow(image_2.reshape(28, 28), cmap='gray')
plt.title("Custom image after normalization and reshaping")
plt.show()

# Make prediction on the custom image
predictions = model.predict(image_2)
predicted_label = np.argmax(predictions, axis=1)

print("Predicted number for mala is :", predicted_label)



































