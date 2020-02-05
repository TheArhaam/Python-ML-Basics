# LOAD & SAVE MODELS

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

arr1 = np.array([1,2,3,4,5,6,7,8,9,10])
arr2 = np.array([6,7,8,9,10,11,12,13,14,15])

model.fit(arr1,arr2,epochs=200)

res = model.predict([20])[0][0].astype(float)

print(res)