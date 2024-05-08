import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('age_prediction_model.h5')

# Function to preprocess image
def preprocess_image(image):
    image = tf.image.resize(image, (64, 64))  # Resize image to match model input shape
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to open file dialog, predict age, and display image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = np.array(img)
        img = preprocess_image(img)
        age = predict_age(img)
        display_image(img, age)

# Function to predict age from image
def predict_age(img):
    age = model.predict(img)
    return age[0][0]

# Function to display selected image and predicted age
def display_image(image, age):
    img = Image.fromarray(np.uint8(image[0] * 255))  # Convert normalized image to uint8
    img = img.resize((200, 200))  # Resize image for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.img = img_tk  # Save reference to avoid garbage collection
    image_label.config(image=img_tk)
    age_label.config(text="Predicted Age: {:.2f}".format(age))

# Create main application window
root = tk.Tk()
root.title("Age Predictor")

# Title label
title_label = ttk.Label(root, text="Welcome to Age Predictor", font=("Arial", 20))
title_label.pack(pady=10)

# Create a frame to hold the image with border
image_frame = ttk.Frame(root, borderwidth=2, relief="groove")
image_frame.pack(pady=10)

# Label to display selected image
image_label = ttk.Label(image_frame)
image_label.pack(padx=10, pady=10)

# Button to open file dialog
open_button = ttk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=5)

# Label to display predicted age
age_label = ttk.Label(root, text="Predicted Age: -", font=("Arial", 14))
age_label.pack(pady=5)

root.mainloop()
