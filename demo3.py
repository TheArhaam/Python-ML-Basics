# Introducing convolutional neural networks
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images / 255.0

# Sequential: That defines a SEQUENCE of layers in the neural network
model = keras.Sequential([
    # Generate 64 filters
    # Every epoch it'll figure out which filters gave the best features(signals) to match the image to the label
    keras.layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    # For compressing the image and enhancing the features
    keras.layers.MaxPool2D(2,2),

    # Stacking convolutional layers on top of each other to break down the image 
    # This will help in finding important features i guess
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    
    # Flatten: Our images are squares, Flatten just takes those squares and turns them into a 1 dimensional set.
    keras.layers.Flatten(),
    
    # Dense: Adds a layer of neurons
    # Each layer of neurons need an activation function to tell them what to do
    # Relu effectively means "If X>0 return X, else return 0" -- so only Positive values are returned
    # Softmax takes a set of values, and effectively picks the biggest one
    # 128 is the number of function
    # 10 means the output will be between 1-10
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))