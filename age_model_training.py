
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models

import requests
from io import BytesIO

# Load dataset
data_path = '/content/drive/MyDrive/IMDB_CSV/gt.csv'
data = pd.read_csv(data_path)

# Preprocess images
image_data = []
labels = []

# Loop through the data and preprocess images
for index, row in data.iterrows():
    img_url = row['label']
    age = row['score']

    try:
        # Load image from URL
        response = requests.get(img_url)
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)

        if img is not None:
            # Resize image to a fixed size
            img = cv2.resize(img, (64, 64))

            # Convert grayscale images to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Append image and label to lists
            image_data.append(img)
            labels.append(age)
        else:
            print(f"Error loading image: {img_url}")
    except Exception as e:
        print(f"Error downloading image: {img_url}, {e}")

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Normalize images
image_data = image_data / 255.0

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

from keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the number of models in the ensemble
num_models = 5
models = []

# Create instances of the CNN model
for _ in range(num_models):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    models.append(model)

batch_size = 32
epochs = 30

# Train each model on different subsets of the training data
for i, model in enumerate(models):
    print(f"Training Model {i + 1}")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test)
    )

# Predict age labels for test data
y_pred = model.predict(X_test)

# Calculate mean absolute error (MAE)
mae = np.mean(np.abs(y_pred - y_test))

# Calculate accuracy (percentage of correct predictions within a tolerance)
tolerance = 5  # Define tolerance for correct predictions
correct_predictions = np.sum(np.abs(y_pred - y_test) <= tolerance)
total_predictions = len(y_test)
accuracy = (correct_predictions / total_predictions) * 100

print("Test Mean Absolute Error:", mae)
print("Test Accuracy:", accuracy, "%")

# Calculate the mean absolute error (MAE)
mae = np.mean(np.abs(y_pred.flatten() - y_test))

# Print actual and adjusted predicted age labels for a few samples
num_samples = 10  # Number of samples to display
for i in range(num_samples):
    print("Sample", i+1)
    print("Actual Age:", y_test[i])
    adjusted_age = round(y_pred[i][0] + mae)  # Adjusted predicted age
    print("Adjusted Predicted Age:", adjusted_age)
    print()

# Save the model
model.save("age_prediction_model.h5")

# Print confirmation message
print("Model saved successfully!")