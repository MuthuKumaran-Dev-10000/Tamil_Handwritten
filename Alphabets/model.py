import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import emnist  # Ensure to install the emnist package

# Define the data augmentation
def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image

# Preprocess the data
def preprocess(image, label):
    label = tf.clip_by_value(label, 0, 25)  # Ensure label is valid
    image = tf.image.resize(image, [28, 28])
    
    if image.shape[-1] == 3:  # Convert to grayscale if RGB
        image = tf.image.rgb_to_grayscale(image)

    image = data_augmentation(image)
    return image, label  # Return the processed image and label

# Prepare datasets
def prepare_datasets(ds_train, ds_test):
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    return ds_train, ds_test

# Define your model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(26, activation='softmax')  # Output layer for 26 classes
    ])
    return model

# Load your dataset
def load_dataset():
    (x_train, y_train), (x_test, y_test) = emnist.extract_training_samples('letters')
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension
    return (x_train, y_train - 1), (x_test, y_test - 1)  # Adjust labels

# Main function
def main():
    (x_train, y_train), (x_test, y_test) = load_dataset()

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_train, ds_test = prepare_datasets(ds_train, ds_test)

    model = create_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(ds_train, epochs=10, validation_data=ds_test)

    model.save('trained_model.h5')
    print("Model saved as 'trained_model.h5'.")

if __name__ == "__main__":
    main()
