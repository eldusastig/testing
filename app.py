import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

# Define the model file path
model_path = 'model1.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the trained model
    model = load_model(model_path)

    # Define the class labels
    class_labels = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']

    # Function to predict the class of an image
    def predict_image(img_path, model):
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        return predicted_class, np.max(predictions)

    # Streamlit app
    st.title("Bean Leaf Lesion Classification (Angular, Bean Rust and Healthy)")
    st.write("Upload an image to classify")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict the image
        label, confidence = predict_image("uploaded_image.jpg", model)
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")
