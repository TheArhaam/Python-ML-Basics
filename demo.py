# The Hello World of TensorFlow ML

import tensorflow as tf
import numpy as np
from tensorflow import keras

#  It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# code to compile our neural network
# we have to specify 2 functions, a `loss` and an `optimizer`.
# The `loss` function measures the guessed answers against the known correct answers and measures how well or how badly it did.
# the model uses the optimizer function to make another guess
# The model will repeat this for the number of epochs
# `mean squared error` for the loss and `stochastic gradient descent` (sgd) for the optimizer.
model.compile(optimizer='sgd',loss='mean_squared_error')

# `numpy` provides lots of array type data structures that are a defacto standard way of providing data
# We declare that we want to use these by specifying the values as an array in numpy using `np.array[]`
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# The process of training the neural network, where it 'learns' the relationship between the Xs and Ys 
# is in the model.fit call. This is where it will go through the loop we spoke about before making a guess, 
# measuring how the loss, using the optimizer to make another guess etc. 
# It will do it for the number of epochs you specify.
model.fit(xs, ys, epochs=100)   

# You can use the model.predict method to have it figure out the Y for a previously unknown X.
print(model.predict([10.0]))

# Neural networks deal with probabilities, so given the data that we fed the NN with, 
# it calculated that there is a very high probability that the relationship between X and Y is Y=3X+1, 
# but with only 6 data points we can't know for sure. As a result, the result for 10 is very close to 31, 
# but not necessarily 31.
