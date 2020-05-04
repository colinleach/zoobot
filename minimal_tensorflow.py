import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':

    print('TF version: ', tf.__version__)

    print('Using GPU: ', tf.config.experimental.list_physical_devices('GPU'))

    # https://github.com/tensorflow/tensorflow/issues/24496
    # this is required for models more complicated than just dense layers, on my 1650 RTX card
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #       tf.config.experimental.set_memory_growth(gpu, True)

    # copied from https://keras.io/getting-started/sequential-model-guide/

    # Generate dummy data
    x_train = np.random.random((100, 100, 100, 3))
    y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

    model = tf.keras.Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    callbacks = [tf.keras.callbacks.TensorBoard(
        log_dir='temp', profile_batch=0  # or 2 to cause CUPTI permissions error
    )]

    model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=callbacks)

    print('Success - exiting')