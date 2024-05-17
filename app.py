import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import os

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    img = np.array(img)
    img = img.reshape(1, 128, 128, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class, np.max(predictions)

# Load the model
model = load_model('model3.h5')

# Define the class labels
class_labels = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']

# Streamlit UI
def main():
    # Sidebar
    st.sidebar.title("TEAM 8 Model Deployment in the Cloud")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About the Project"])

    if page == "Home":
        # Title
        st.title("Application")

        # Main page content
        st.write("Welcome to the Rose Leaf Classification App! This app uses a Convolutional Neural Network (CNN) model to classify images")
        st.write("Upload an rose image and the app would classify if it's Healthy, Rose Rust or have Rose Slug Sawfly damage")

        # List of health categories
        health_categories = [
            "Healthy",
            "Rose Rust",
            "Rose Slug Sawfly damage"
        ]

        st.write("Health Categories:", health_categories)

    elif page == "Prediction":
        # Prediction page
        st.title("Model Prediction")
        st.write("Upload an image to predict the condition of the leaf.")

        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=300, caption='Uploaded Image')
            if st.button("Predict"):
                st.write("Predicting...")
                predicted_health, confidence = predict_image(test_image, model)
                st.write(f"Predicted Condition Category: {predicted_health}")
                st.write(f"Confidence: {confidence:.2f}")

    elif page == "About the Project":
        # About the project
        st.title("About the Project")
        st.write("""
        This Streamlit app uses a Convolutional Neural Network (CNN) model to classify different condition categories.  
        """)
        st.write("Developed by: Team 8 (CPE32S9)")
        st.write("- Duque, Jethro")
        st.write("- Natiola, Henry Jay")

if __name__ == "__main__":
    main()
