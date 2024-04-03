import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Define function to create and train the model
def train_model(batch_size, epochs):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=0)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return test_acc

# Train the model with different batch sizes and epochs
batch_sizes = [32, 64, 128]
epochs_list = [10, 20, 30]#change if you want here


results = {}

for batch_size in batch_sizes:
    for epochs in epochs_list:
        acc = train_model(batch_size, epochs)
        results[(batch_size, epochs)] = acc

# Tabulate the results
print("Batch Size\tEpochs\tAccuracy")
for key, value in results.items():
    print(f"{key[0]}\t\t{key[1]}\t{value}")