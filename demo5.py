# TRIAL & ERROR FOR INPUT_SHAPE
# Figured out input_shape a bit
# Trained a model for multiplication table of 2 with input_shape = 2

import tensorflow as tf
import numpy as np
from tensorflow import keras
from numpy import ndarray

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[2])])

model.compile(optimizer='sgd',loss='mean_squared_error')

num1 = np.array([[2,1],[2,2],[2,3],[2,4],[2,5]])
num2 = np.array([[2],[4],[6],[8],[10]])

model.fit(num1,num2,epochs=1000)

res = model.predict([[2,20]])

print('ROUNDED PREDICTION= '+str(round(res[0,0].astype(float))))
