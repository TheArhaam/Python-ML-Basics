# LOAD & SAVE MODELS

import tensorflow as tf
import numpy as np
from tensorflow import keras

try:
    # Loading the model from memory if found
    model = keras.models.load_model('demo6model.h5')
except Exception as e:
    # Creating a new model if model not found
    print('Model not found, creating new model.')
    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

arr1 = np.array([1,2,3,4,5,6,7,8,9,10])
arr2 = np.array([6,7,8,9,10,11,12,13,14,15])

# 200 epochs not enough we still get loss: 0.8
# So we will save the model and load it so that we don't have to 
# restart the training from the beginning but where we left off instead
model.fit(arr1,arr2,epochs=200)

res = model.predict([20])[0][0].astype(float)

print(round(res))

# Saving the model
try:
    model.save('demo6model.h5')
    print('Model Saved.')
except Exception as e:
    print('Unable to save model.')

# I ran the program 5 times
# Basically trained the model over the last save itself 5 times
# This allowed me to not need to increase the epochs and restart from the beginning
# I could just pick up from where I last left off
# So once I ran it 5 times I was starting to get the accurate result