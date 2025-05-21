from tkinter.ttk import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0

master = tk.Tk()
master.title("Train logo")
master.geometry("900x300")
master.resizable(False, False)
master.configure(background='wheat1')

def TrainData_Start():
    # Path to your signature dataset folder
    dataset_folder = 'Product'

    # Initialize empty lists to store images and labels
    images = []
    labels = []

    for label in os.listdir(dataset_folder):
        label_folder = os.path.join(dataset_folder, label)

        if os.path.isdir(label_folder):
            for image_file in os.listdir(label_folder):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(label_folder, image_file)
                    images.append(image_path)
                    labels.append(label)

    # Convert lists to NumPy arrays
    X_paths = np.array(images)
    y = np.array(labels)

    # Use LabelEncoder to convert labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Load images from file paths and preprocess using EfficientNetB0 preprocessing
    X = np.array([keras.applications.efficientnet.preprocess_input(image.img_to_array(image.load_img(img_path, target_size=(224, 224)))) for img_path in X_paths])

    # Load the pre-trained EfficientNetB0 model
    base_model = EfficientNetB0(weights='efficientnetb0_notop.h5', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model on top of the pre-trained model
    model = keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(len(np.unique(y_encoded)), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y_encoded, epochs=35)

    # Save the trained model for later verification
    model.save('Detectionmodel_11.h5')
    messagebox.showinfo("Training Complete", "Model has been trained and saved as 'Detectionmodel_1.h5'")

def DetectLogo():
    # Load the saved model
    model = keras.models.load_model('Detectionmodel_11.h5')

    # Open a file dialog to select an image for detection
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class of the image
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)
        
        # Decode the predicted class to the original label
        label_encoder = LabelEncoder()
        label_encoder.fit(os.listdir('Product'))
        predicted_label = label_encoder.inverse_transform(predicted_class)

        #messagebox.showinfo("Detection Result", f"The detected logo is: {predicted_label[0]}")
        messagebox.showinfo("Detection Result", f"The detected logo is: {predicted_label[0]}\nConfidence: {confidence:.2f}")

label = tk.Label(master ,width=40,text = "Train logo",font=("arial italic", 30), bg="medium aquamarine", fg="black").grid(row=0, column=0,columnspan=1)

btnd1 = tk.Button(master,text="Train",font=("arial italic", 15), bg="skyblue2", fg="black",width=25,command=lambda:TrainData_Start()).grid(row=3, column=0,padx=1, pady=20)

btnd2 = tk.Button(master,text="Detect",font=("arial italic", 15), bg="skyblue2", fg="black",width=25,command=DetectLogo).grid(row=4, column=0,padx=1, pady=20)

btnd3 = tk.Button(master,text="Exit",font=("arial italic", 15), bg="tomato", fg="white",width=25,command=master.destroy).grid(row=6, column=0,padx=1, pady=20)

master.mainloop()
