# TRIAL & ERROR FOR INPUT_SHAPE

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=(1,2))])

model.compile(optimizer='sgd',loss='mean_squared_error')

num1 = np.array([[1,1],[1,2]])
num2 = np.array([[1],[2]])

model.fit(num1,num2,epochs=10)
