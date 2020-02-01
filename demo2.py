# Introduction to Computer Vision
import tensorflow as tf
from tensorflow import keras
import numbers
# from tensorflow.optimizers import AdamOptimizer

# The Fashion MNIST data is available directly in the tf.keras datasets API
mnist = keras.datasets.fashion_mnist

# Calling load_data on this object will give you two sets of two lists,
# these will be the training and testing values for the graphics that contain the clothing items and their labels.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images/255.0
test_images = test_images/255.0


# Sequential: That defines a SEQUENCE of layers in the neural network
# Flatten: Our images are squares, Flatten just takes those squares and turns them into a 1 dimensional set.
# Dense: Adds a layer of neurons
# Each layer of neurons need an activation function to tell them what to do
# Relu effectively means "If X>0 return X, else return 0" -- so only Positive values are returned
# Softmax takes a set of values, and effectively picks the biggest one
# Input image size is 28x28
# 128 is the number of function
# 10 means the output will be between 1-10
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20)

# model.evaluate(test_images, test_labels)
# classifications = model.predict(test_images)
# print(test_labels[0])

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
