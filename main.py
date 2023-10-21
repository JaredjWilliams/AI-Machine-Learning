import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add a Dense layer to the model
model.add(Dense(units=1, input_shape=[1]))

# Compile the model with the correct optimizer and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define your input data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model for 500 epochs
model.fit(xs, ys, epochs=500)

# Make a prediction with the trained model
prediction = model.predict([10.0])

# Print the prediction
print("Prediction for 10.0:", prediction)

# Print the learned weights
dense = model.layers[0]  # Retrieve the Dense layer
weights, bias = dense.get_weights()
print("Learned Weights:", weights)
print("Learned Bias:", bias)

