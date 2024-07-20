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


loaded_model = tf.keras.models.load_model('my_cifar10_model.h5')


loss = loaded_model.evaluate(x_test, y_test)
print('Test loss and accuracy:', loss)


sample_image = x_test[22:23]
predictions = loaded_model.predict(sample_image)
print('Predictions:', predictions)

plt.figure(figsize=(10, 10))
plt.imshow(sample_image[0])
plt.title(f'Predicted Class: {np.argmax(predictions)}')
plt.show()
