import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import os

# Define Tamil alphabets
tamil_alphabets = [
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 
    'ஒ', 'ஓ', 'அம்', 'ஃ', 'க்', 'ச்', 'ட்', 'த்', 
    'ப்', 'ர்', 'ன்', 'ய்', 'ள்', 'வ்', 'ஜ்', 'ஜ', 
    'ற்', 'ன்'
]

# Load the dataset
def load_dataset(dataset_dir):
    images = []
    labels = []
    for idx, char in enumerate(tamil_alphabets):
        for i in range(10):  # Load all variations per character
            image_path = os.path.join(dataset_dir, f"{char}_{i}.png")
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                img = img.resize((100, 100))  # Resize to 100x100
                
                # Thresholding
                img = np.array(img)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                
                # Normalize and append
                img = img / 255.0
                images.append(img)
                labels.append(idx)
    
    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images, labels

# Data augmentation function
def augment_image(image):
    # Random rotation
    angle = random.uniform(-15, 15)  # Rotate by a small angle
    image = tf.image.rot90(image, k=random.randint(0, 3))  # Rotate randomly
    image = tf.image.adjust_brightness(image, random.uniform(0.8, 1.2))  # Adjust brightness
    return image

# Build the model
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Load the dataset
images, labels = load_dataset('tamil_alphabets_dataset_with_mistakes')
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Augment the training dataset
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.map(lambda x, y: (augment_image(x), y)).batch(32).repeat()

# Build and compile the model
num_classes = len(tamil_alphabets)
model = build_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, steps_per_epoch=len(x_train) // 32)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save the model
model.save('tamil_character_recognition_model.h5')
print("Model saved successfully.")
