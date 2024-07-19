import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# The MinMaxScaler() takes 2D arrays as inputs , hence we need to reshape

train_input = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0]).reshape(-1, 1)
train_output = np.array([4.0, 8.0, 12.0, 16.0, 24.0, 28.0, 32.0, 36.0, 40.0]).reshape(-1, 1)

test_input = np.array([9.0, 17.0, 25.0, 70.0]).reshape(-1, 1)
test_output = np.array([18.0, 34.0, 50.0, 140.0]).reshape(-1, 1)

#Make sure you fit_transform your training set and transform your test set

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

scaled_train_input = input_scaler.fit_transform(train_input)
scaled_train_output = output_scaler.fit_transform(train_output)

scaled_test_input = input_scaler.transform(test_input)
scaled_test_output = output_scaler.transform(test_output)

# Always set the learning rate to 0.01 as the model finds it hard to learn.

model = keras.Sequential([

    #An activation function is essentially used to introduce a non-linearity in plotting the graph , so that the model can asses the values and start learning

    #Sigmoid - Binary Class Classification
    #Softmax -Multi Class Classification
    keras.layers.Dense(units=32, input_shape=[1], activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1)
])

#An optimizer is used to adjust the weights on a node to minimize loss function
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(scaled_train_input, scaled_train_output, epochs=500)

loss = model.evaluate(scaled_test_input, scaled_test_output)
print("Loss:",loss)

'''
new_input = np.array([10.0]).reshape(-1, 1)
scaled_new_input = input_scaler.transform(new_input)
'''
predictor = model.predict(scaled_test_input)

system_prediction = output_scaler.inverse_transform(predictor)
print("Denormalized Prediction:", denormalized_prediction)

plt.figure(figsize=(10, 6))
plt.plot(test_input, test_output, label='Real Test Outputs', color='red')
plt.plot(test_input, system_prediction, label='Predicted Test Outputs', color='green')
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.grid(True)
plt.legend()
plt.show()
