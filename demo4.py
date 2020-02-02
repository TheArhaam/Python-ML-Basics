# Multiplication table of 2

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

nums1 = np.array([1,2,3,4,5,6,7,8,9,10])
nums2 = np.array([2,4,6,8,10,12,14,16,18,20])

model.fit(nums1,nums2,epochs=500)

print(model.predict([11]))